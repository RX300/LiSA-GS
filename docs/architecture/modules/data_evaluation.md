# 数据与评价

`lisags/data/dataset.py` 原样保留 PNG/EXR 解码、长边缩放、相机转换和 SSS 光坐标契约，仅修改项目名称。
训练只用 `transforms_train.json`。详细数据契约见 [data_contract.md](../../data_contract.md)。

`lisags/evaluation/evaluator.py` 支持 fit/train/validation/test，加载最终状态后调用与训练相同的渲染函数。
PNG 先对前景做 gamma 编码再 alpha 合成；HDR 对完整图像编码。
主指标使用截断后 8-bit 量化 RGB：PSNR、零填充 11×11 SSIM（sigma=1.5）、VGG LPIPS（输入 [-1,1]）。
测试帧使用原始相机和灯位；训练帧才应用保存的训练校正。

原项目的 shift-align 和测试标定诊断仍可显式调用，输出另列，不能作为官方主指标。
`--save-all` 保存全部 GT/预测横向对照图，`--highlights` 保留论文中的高光代理统计。
