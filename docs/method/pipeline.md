# 数据到输出

1. 只读取官方训练帧及其相机、点光源、alpha；训练相机确定场景归一化和光强归一化。
2. 用最多 32 个训练轮廓采样自身 3DGS 种子，初始化高斯、SVBRDF 网络和学习法线。保留原来的随机数调用顺序。
3. 前 8k 步只训练几何和预热外观；最后三分之一预热将目标编码平滑转为线性域。
4. 继续 22k 步线性域逐高斯着色的联合训练，总共 30k；几何增密、裁剪、opacity 重置及学习率与原工程一致。
5. 从 `geometry/last.pt` 初始化像素着色外观精修，冻结 means/scales/quats/opacities 及训练相机标定；
   再训练 8k 步外观。新的优化器与原始阶段交接一致，不把它当作优化器续训。
6. 从 `appearance/last.pt` 渲染官方 test。使用原始测试标定，无测试时拟合；输出 PNG 对照图与 PSNR/SSIM/LPIPS。

Real_NRHints 在训练期间采用旋转相机校正与平移 gauge；Synthetic_GS3 为白背景；
Synthetic_SSS-GS 为 256 分辨率、单位光强，点光源使用世界坐标。
精确参数以 [staged.json](../../configs/staged.json) 为准。
