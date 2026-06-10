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
