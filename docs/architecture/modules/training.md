# 训练模块

`lisags/training/options.py` 保存单阶段参数及原有适用检查；最终复现实验统一使用 `configs/staged.json`。
`initialization.py` 管理训练帧、轮廓种子、场景归一化和 checkpoint 初始化。
构造高斯/网络再加载权重的顺序会消费 RNG，因此提取时刻意保留。

`schedule.py` 保留辐射目标过渡、geometry_only 开关和学习率公式；
LiSA 预热后的 base 已经是带光照归一化的反射响应，交接时不重置它。
`source.py` 恢复原模型状态，并让固定几何阶段继承相机标定和 shadow/port 开关。
`pose.py` 与 `lisags/geometry/cameras.py` 保留相机旋转优化、平移 gauge、可选光位置及光强拟合。
最终配方只开启真实数据的训练相机优化。

最终损失顺序：0.8 L1、0.2 SSIM、alpha L1、feature 正则、相机正则。
前 2k 步的 foreground weighting 和几何梯度辅助 pass 均保持原实现。
