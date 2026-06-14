# KITTI Depth Refine

## 背景
* 参考kitti_depth_inpaint.md，当前已经完成基于稀疏点云的深度补全核心功能depth_inpainter.py
* 但是稀疏点云可能会有高反膨胀、传感器标定导致的偏移等问题

## 目标
* 基于depth_inpainter.py开发一个depth_refiner,使生成的稠密点云中尽量减少高反、偏移带来的影响

## 设计理念
* 稀疏点云和条件深度生成模型，可以看成两种不同的regular，需要深度整合并相互校验

## 方案实施

### 先解决点云膨胀问题

#### Step 1

##### 目标
* 以kitti depth为基础，构造膨胀的深度数据

##### 实施方案
* 在kitti_utils.py中增加一个对深度图做膨胀操作的函数
* 在局部确保前景的膨胀覆盖背景的膨胀
* 使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png作为图片和深度值输入
* 编写一个refine_step1.py进行测试，保存膨胀前和膨胀后的rgb,depth叠加图像，图像保存到refine_step1/


#### Step 2

#### 现状
* 如果直接输入膨胀的点云，经过depth_inpainter不能很好的精细化

#### 尝试
* 考虑对输入depth_inpainter的点云做 14x14 patch的降采样，即 14x14 范围内只随机保留一个点，这样应该会增强深度生成模型* 使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和膨胀后的depth refine_step1/inflated_depth.png来测试的regular效果
* 参考test_step6.py，实现上述功能，保存的图和点云也参考test_step6.py
* 创建refine_step2.py,实现上述功能，结果保存到refine_step2/目录下
* 使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和膨胀后的depth refine_step1/inflated_depth.png来测试


#### Step 3

#### 现状
* 确实修复了膨胀的问题，但是depth_inpainter输出的物体位置与原始kitti depth点云存在较大的偏差

#### 可能的原因
* 过于稀疏的点云作为depth_inpainter的输入，导致ransac可用点云变少

#### 新的尝试
* depth_inpainter还是输入完整的点云
* 只在ppd/models/depth_inpainter.py：110降低know_depth的点数
* 为DepthInpaintPipeline增加一个patch下采样参数，默认设置为7x7，即7x7范围内随机保留一个有效深度值
* 创建refine_step3.py,实现上述功能，结果保存到refine_step3/目录下
* 使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和膨胀后的depth refine_step1/inflated_depth.png来测试
* 保存：输入overlay图像，inpaint overlay图像，inpaint点云，输入左右可视化图像，inpaint左右可视化图像

#### Step 4

#### 现状
* 当使用path_size=2后发现，depth_inpainter输出的物体位置与原始kitti depth点云存在较大的偏差，看上去问题出在尺度对齐上

### 问题排查

#### 排查1：校验下采样点云
* 实现patch下采样点云保存
* 我会查看点云是否有问题

#### 排查2：校验实际进入lanpaint inpainter的深度图
* 增加一个debug标签，在ppd/models/depth_inpainter.py：112行以后增加resize之后的rgb+depth overlay图像，叠加时使用point size=1
* 我会查看overlay不同深度输入情况下的overlay有什么差异

##### 现状
* 我执行了refine_step3.py，发现产生的resize之后的rgb+depth overlay图像，下半部分是没有深度点的

##### 执行
* 请你排查当前问题，可以在depth_inpainter.py增加调试代码

#### 发现问题
* depth_inpainter.py：128-132行是在ppd log空间进行下采样的，这导致用0判断mask出现问题

#### 修复
* 我们可以对mask进行下采样，然后可以很容易的得到原始空间和ppd空间的深度值
* 基于上面的描述对depth_inpainter.py进行修改

#### Step 5
* 删除DepthInpaintPipeline中的known_patch_size参数


#### Step 6

##### 目标：基于泊松深度补全的demo
* 可以让ppd输出一个相对深度，然后根据稀疏depth，通过ransac恢复绝对深度
* 然后利用ppd_metric_depth和稀疏depth，利用泊松深度补全，得到最终的深度
* 只编写一个refine_step7.py文件，实现上述功能. 生成点云和overlay的深度图到refine_step7/目录下
* 使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png作为图片和深度值输入

##### 参考
* /home/fanguedong/LLM_Wiki/Visual_3D/wiki/overview.md
* /home/fanguedong/Code/ViGeo/depth_refine.py
* /home/fanguedong/Code/ViGeo/

##### 补充改动 1
* 考虑在泊松补全前对稀疏深度做过滤，参考/home/fanguedong/Code/ViGeo/depth_refine.py:68

##### 补充改动 2
* 把refine_step7.py重命名为refine_step6.py，refine_step7/重命名为refine_step6/
* refine_step7.py里面摘抄了很多/home/fanguedong/Code/ViGeo/里面的代码，能直接import的话就直接import，目前不重复实现

##### 补充改动 3
* 之前我们一直在尝试如果输入的点云有膨胀，我们的depth refine算法能否有效处理，目前看这个新方案很有希望
* 增加一个可选输入参数，用来膨胀输入的稀疏点云，这样我们可以评判如果点云膨胀较多的情况下效果如何

#### Step 7

##### 目标：点云过滤的depth inpaint
* 参考refine_step6.py，编写refine_step7.py，使用DepthInpaintPipeline完成深度补全代替 **4. Poisson completion**，其他保持不变
* 对保存目录名进行对应修改
* 使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png作为图片和深度值输入
* 进行简单的测试

#### Topic 1

##### 现象
* 我发现通过inpaint方式得到的点云偶尔会在一些地方出现明显的块状的异常深度，结合kitti_depth_inpaint.md

##### 要求
* 总结下可能的原因
* 给出3个可能的解决方案，按重要性排序

##### 可能的原因
* 原因1：RANSAC尺度标定被离群点污染，已知/未知区域尺度衔接处出现台阶（最重要，已被多尺度过滤覆盖）
* 原因2：PPD结构先验在KITTI盲区产生幻觉，LanPaint FLD强化这种幻觉形成空间连贯但数值偏移的块状区域
* 原因3：已知/未知边界硬过渡，二值mask + λ=16使约束强度差17倍，稀疏点边缘产生深度台阶

##### 解决方案（按重要性排序）
* 方案1：稀疏点云多尺度过滤（已由refine_step7.py实现）
* 方案2：PPD置信度加权 + Poisson融合，低置信区用Poisson替代LanPaint
* 方案3：软边界mask + 降λ + 增采样步数

##### 发现：外层early stop未实现
* 原始LanPaint代码（nodes.py:180）有外层early stop机制，默认在最后1步跳过FLD（n_steps=0），退化为标准denoise
* 我们的lanpaint_inpainter.py所有步骤都跑FLD，包括最后一步
* 理论分析：SHO的有效参数（Gamma·t=225, A·t=3.4, Delta≈0.94）是abt无关的常数，SHO本身在abt→1时稳定
* 真正影响：最后一步的FLD随机采样扰动无法被后续denoise平滑，直接留在最终输出中，可能形成局部的深度跳变
* 已在lanpaint_inpainter.py实现该机制，可通过early_stop参数控制（默认1）

##### 现状
* 设置early_stop=1导致最后一步稀疏深度无法被利用到，导致输出的点云与稀疏深度点云差别变大，所以我们还是把early_stop全部默认设置为0


#### Step 8

##### 目标
* 优化下refine_step7.py的调用流程，现在会加载两次相同的ppd模型
* 第一次创建ppd模型的位置，可以使用DepthInpaintPipeline内部的ppd模型，这样就不用加载两次了
* 直接修改refine_step7.py，完成后进行简单测试


#### Step 9

##### 目标
* 编写一个evaluate_kitti.py用来批量测试深度补全效果（使用refine_step7.py中的补全方案）

##### 参考
* /home/fanguedong/Code/ViGeo/evaluate.md（数据路径、指标说明、中间结果保存等）
* /home/fanguedong/Code/ViGeo/evaluate_kitti.py

##### 测试
* 编码完成后使用10个样本的mini dataset进行简单的测试

#### Step 10

##### 现状
* evaluate_kitti.py跑10个样本后发现，预测的点云尺度和gt存在比较大的差别

##### Debug
* 从refine_step7.py复制并创建refine_step10.py
* 使用eval_kitti_output/2011_09_26_drive_0002_sync_image_0000000032_image_03对应的样本，执行refine_step10.py（需要修改输入图片，稀疏深度和内参），结果保存到refine_step10/目录下
* 尽量少修改完成上面的需求，这样我可以通过compare refine_step7.py和refine_step10.py保证代码不存在大的问题

##### 发现问题
* `filtered_overlay.png`（PPD分辨率，过滤后）有很多点，但 `known_depth_overlay.png`（pipeline内部，实际进入LanPaint的深度）点非常少
* 大量稀疏深度点在 PPD→原始→PPD 的分辨率往返中丢失

##### 根因
* 代码用 `sparse_depth * (filtered_mask_orig > 0)` 将过滤mask映射回原始分辨率
* `sparse_depth`（velodyne）的点是原始像素位置，`filtered_mask_orig` 的True像素位置由 NEAREST resize 的逆映射决定
* 两者因 resize 往返的 ±1 像素偏移不对齐，导致 `sparse_depth[h,w] * mask[h,w]` 几乎全为0
* 例如：原始 h=100 → PPD rh=136 → 逆映射回 h'=99（非 h=100），velodyne 点被错误置零

##### 修复
* 不再用 mask 去筛选原始 velodyne 点，而是直接把 PPD 分辨率的过滤后深度 resize 回原始分辨率
* 这样深度值和位置一起迁移，pipeline 内部再 resize 回 PPD 时位置一致
* 修复涉及3个文件：refine_step10.py、refine_step7.py、evaluate_kitti.py

##### 修复效果（sample 0032）

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| MAE | 1.3409 m | 0.5481 m | -59% |
| RMSE | 2.6043 m | 1.8031 m | -31% |
| δ₁ | 0.9875 | 0.9940 | +0.7% |
