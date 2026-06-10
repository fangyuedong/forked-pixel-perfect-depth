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


## Step 6

### 当前情况
* 在Step 5中，我们已经修复了问题，并且通过test_step4.py完成了验证

### 抽象depth inpaint pipeline核心实现
* 将test_step4.py中的核心实现抽象为一个深度补全类
* 该类的核心功能是：当调用__call__时，输入图像和稀疏的带尺度深度，输出补全的带尺度深度图
* 编写test_step6.py，调用该类，使用/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png和/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png作为图片和深度值输入，可以输出与test_step4.py完全一致的有尺度深度图和点云结果
* 通过md5sum校验结果是否一致，为此需要考虑使用固定的随机种子
* 保持代码简洁,产生的数据和结果都放到step6/目录下
* 为test_step6.py增加图片和深度gt配置参数，方便进行批量测试

## Topic: Lanpaint参数

### Lanpaint参数的作用

LanPaint的核心算法是Fast Langevin Dynamics (FLD)，在每个扩散步内进行N次Langevin迭代来逼近精确的条件分布。FLD的动力学由Stochastic Harmonic Oscillator (SHO)描述：

```
dx = q dt
dq = -Γ A x dt + Γ C dt + Γ D dw - Γ q dt
```

其中x是位置(深度潜变量)，q是速度，Γ是摩擦，A是谐势强度，C是常力，D是噪声幅度。

#### fld_steps (NSteps, 默认5)

每个扩散步内FLD的迭代次数N。论文定理3.2证明：当N→∞时，FLD的输出精确收敛到条件分布p(x₀|x_t, y)。

* N=0：不执行FLD，退化为标准DDPM+replace（即只做replace step和标准denoise）
* N越大，条件约束越精确，已知区域保留越好
* 实际中N=5已能取得较好效果，N=10进一步提升但边际收益递减
* 计算开销与N线性正比（每次FLD迭代需要一次DiT前向推理）

#### fld_step_size (StepSize η, 默认0.2)

Langevin动力学的基础步长η。实际步长会随时间自适应调整：`η_adaptive = η × (1 - ᾱ_t)`

* `ᾱ_t → 1`（接近干净图像，小t）：步长→0，小步精细调整
* `ᾱ_t → 0`（纯噪声，大t）：步长→η，大步快速移动

η控制FLD每次迭代的移动幅度：
* η太大：迭代不稳定，可能震荡或发散
* η太小：收敛慢，需要更多迭代才能逼近条件分布
* η=0.2是论文推荐值，在稳定性和效率之间取得平衡

#### fld_lambda (Lambda λ, 默认16.0)

BiG Score（Bidirectional Guidance Score）中已知区域的条件强度。BiG Score定义（lanpaint.py:140）：

```
score_known = -(1+λ)(x_t - y) + λ(x_t - x̂₀)
```

其中y是已知深度，x̂₀是模型预测。这等价于：
* `(1+λ)(y - x_t)`：将x_t拉向已知值的力，强度为(1+λ)
* `λ(x̂₀ - x_t)`：模型预测方向的力

λ通过影响谐势强度A_y来间接控制FLD动力学（lanpaint.py:276）：
```
A_y = (1 + λ) / (1 - ᾱ_t)
```

* λ越大：A_y越大，已知区域的"弹簧"越硬，x_t被更强烈地拉向已知值y
* λ太小：已知区域约束不足，已知深度值会被扩散过程扰动
* λ太大：可能导致数值不稳定（A_y过大使SHO振荡）
* λ=16.0是论文推荐值，对应将已知值的约束强度设为模型预测的16倍

#### fld_friction (Friction Γ, 默认15.0)

Langevin动力学的摩擦系数Γ。控制SHO中的阻尼程度（lanpaint.py:269-270）：

```
Γ̂ = Γ² × η × σ / 0.1 / 2
```

Γ影响的是速度项q的衰减速率：
* Γ大：强阻尼，速度快速衰减，系统趋向overdamped（一阶Langevin），轨迹更稳定但探索能力弱
* Γ小：弱阻尼，速度持续更久，系统趋向underdamped，探索能力强但可能不稳定
* Γ→∞：退化为overdamped Langevin（忽略速度项）
* Γ=15.0提供适中的阻尼，在二阶Langevin动力学中平衡稳定性和探索效率

#### 参数之间的关系

这些参数协同工作：
* **η和Γ** 共同决定SHO的有效时间步长：`dt = 2η`，摩擦为`Γ²η/0.2`
* **η和N** 决定总的Langevin积分时间：总时间≈ N × dt = 2Nη
* **λ** 单独控制已知/未知区域的力度比例
* 增大N或η都能增加Langevin的"积分量"，但N通过更多小步实现（稳定），η通过更大单步实现（高效但风险高）

#### 典型调参建议

| 目标 | 调整 |
|------|------|
| 更好保留已知区域 | 增大λ (16→32) 或增大N (5→10) |
| 更快推理 | 减小N (5→0或3)，减小sampling_steps |
| 数值不稳定 | 减小η (0.2→0.1)，增大Γ (15→20) |
| 已知区域过度锐化 | 减小λ (16→8) |

## Step 7

### 背景
* 我仔细阅读了你Step 5做的改动，发现**Bug 3: replace_step每次生成新噪声**，有些疑问
* 我阅读了lanpaint.py,发现LanPaint调用一次__call__应该是调用了一次模型原本的去噪步+一轮lanpaint迭代
* 所以lanpaint.py:44-45生成的噪声只作用给了out_loop的一轮迭代

### 测试
* 为了验证上面的观点，我们需要进行对比测试
* 从kitti depth数据中随机采样10个样本，跑一下test_step6.py，结果放到step6/no_replace_noise
* 修改下replace_noise逻辑，改成每轮out_loop都随机生成noise. 使用上述相同样本跑下test_step6.py，结果放到step6/replace_noise
* 10个样本从kitti depth目录下随机挑，不要只挑一个序列的

### 实现要求
* 这次我们只最小化修改已有python脚本，实现上述测试，不增加新的python脚本


## Step 8

### 目前情况
* 看上去两个版本没有明显差别

### 实现要求
* 相关py文件全部改成replace_noise方式，删除no_replace_noise选项