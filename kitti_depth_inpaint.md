# Kitti Depth Inpaint

## Step1

### 方案
* 利用PixelPerfectDepth和LanPaintInpainter对kitti depth的稀疏深度数据进行补全
* 新增一个py文件实现上述功能
* 保持代码简洁

### 测试
使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png作为图片和深度值输入，输出补全后的深度图、点云、深度图叠加原图像


## Step2

### 需求
* 在kitti_utils.py中增加保存真值点云（带颜色）的功能


## Step3

### 当前情况
* inpaint的点云会出现近处畸变的情况，但是远处都还不错

### 我的猜测
* 怀疑是ppd的训练分布和kitti的分布不是很匹配

### 验证方式
* 用run_point_cloud.py跑出结合moge尺度化的ppd预测的深度图，depth_metric_ppd
* 根据kitti depth对应的深度图对depth_ppd做mask，得到稀疏程度和kitti一致但是深度分布是ppd的深度图depth_metric_ppd_masked，并且保存为kitti depth相同的格式
* 用run_kitti_inpaint.py加载depth_metric_ppd_masked和对应的rgb图片，跑出深度图和点云

### 实现方式
* 编写一个test_step3.py脚本,整合上述过程，并且跑出结果
* 保持代码简洁,产生的数据和结果都放到step3/目录下

## Step 4

## 当前情况
* 问题并没有解决，可能还是当前使用的归一化的方式没有归一化到ddp需要的范围

### 验证方式
* 用run_point_cloud.py跑出ppd预测的相对深度图ppd_relative_depth, 利用kitti depth的gt，计算ransac系数a, b. (log(gt+1)=a*ppd_relative_depth+b)
* 将归一化的gt深度： （log(gt+1)/a - b）- 0.5 作为 inpaint的已知深度，得到inpaint的相对深度ppd_inpaint_relative_depth
* ppd_inpaint_relative_depth 再和 log（gt+1） 做ransac，得到 ppd_inpaint_metric_depth
* 跑出ppd_inpaint_metric_depth的深度图和点云

### 实现方式
* 编写一个test_step4.py脚本,整合上述过程，并且跑出结果
* 保持代码简洁,产生的数据和结果都放到step4/目录下

## Step 5

### 当前情况
* 查看Step 4输出的点云后，发现点云在近处依然有很大的畸变

### 检查Lanpaint核心实现是否正确
* 对照原始代码仓/home/fanguedong/Code/LanPaint
* 对照原始论文/home/fanguedong/Code/forked-pixel-perfect-depth/LanPaint/arXiv-2502.03491v3
* 仔细阅读代码，自行探索可能存在问题的地方，尝试进行问题修复，问题修复后执行test_step4.py请我确认效果

### 发现的Bug

对比原始LanPaint代码(lanpaint.py)和当前实现(lanpaint_inpainter.py)，发现3个关键Bug：

#### Bug 1: compute_score未将x_t从internal format转回model format再调用DiT
* **问题**: FLD循环中x_t处于internal format (x_internal = x_model * (sqrt(abt) + sqrt(1-abt)))，但compute_score直接将internal format的x_t传给DiT模型。DiT期望接收model format输入，导致模型预测不准。
* **原始代码** (lanpaint.py:129): `x = x_t / (abt**0.5 + (1-abt)**0.5)` 先转回model format再调用模型
* **修复**: compute_score中先计算abt和转换因子，将x_t转为model format后再拼接输入传给DiT

#### Bug 2: 缺少最终替换步骤
* **问题**: inpaint()方法最后直接返回x_t，没有将已知像素恢复为原始值。扩散过程会轻微扰动已知区域的深度值，近处物体对深度误差敏感，导致近处畸变。
* **原始代码** (lanpaint.py:120): `out = out * (1-latent_mask) + self.latent_image * latent_mask` 在最终输出时精确恢复已知像素
* **修复**: 在inpaint()末尾添加 `x_t = x_t * edge_mask + known_depth * (1 - edge_mask)`

#### Bug 3: replace_step每次生成新噪声
* **问题**: replace_step()每步调用`torch.randn_like(x)`生成新的随机噪声，导致每步已知区域的替换值不同，引入不一致性。
* **原始代码** (lanpaint.py:44-45): 在`__call__`中一次性生成并存储噪声`self.noise`，后续所有步骤复用同一份噪声
* **修复**: 在inpaint()开头生成一次噪声存为`self._replace_noise`，replace_step()中复用

### 修复效果验证

修复前后通过test_step4.py对比（RANSAC a=2.95, b=1.25）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 反变换RANSAC a2 | 3.03 | 2.95 (与a一致) |
| 反变换RANSAC b2 | 1.23 | 1.25 (与b一致) |
| 已知像素均值 | 14.35m | 14.24m |
| KITTI GT均值 | 14.20m | 14.20m |

修复后正/反RANSAC系数完全一致，证明已知像素被完美保留，不再被扩散过程扰动。
