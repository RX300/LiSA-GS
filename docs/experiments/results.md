# 实验结果

本轮独立项目重跑已完成 18/18 场景、5691 张官方 test；Pikachu 补跑已完成。
与四个 baseline 的同场景对比见 [本轮结果表](baseline_comparison.md)、[逐场景 CSV](baseline_comparison.csv)。
以下为之前的迁移与历史结果记录。

## 历史结果与迁移核对

已有最终 LiSA-staged 实验覆盖 18 个场景、5691 张官方 test。
原报告场景等权 PSNR / SSIM / LPIPS 为 **33.4722 / 0.95199 / 0.05636**。
这是原工程已完成的 38k 单种子结果，本次没有重跑完整训练，也不产生新的质量结论。
另有统一 8-bit GT 重评价（common_ground_truth），与此处的原生评价属于不同协议，不能混用。

最终权重、原 config、split、history 和完整 test 对照图/指标位于：
`runs/lisa_staged_full_dataset_20261004/<family>/<scene>/appearance/`。
几何阶段仅保留训练元数据，未复制中间模型和旧方法源码归档。
补充实验数表保存在 [supplementary/](supplementary/)，论文只需的少量消融/损失域图片也已复制。

迁移数值核对记录在 [migration_validation.json](migration_validation.json)。
18 场景原分辨率单帧 RGB/alpha/target 最大绝对差均为 0；模型构造权重和 RNG 状态逐位一致。
这支持代码提取的一致性，不等同于重新训练全部 18 场景的统计性能验证。

短程实测使用真实 Cat 的四帧、32 分辨率、128 个初始高斯：8 步几何/联合训练（其中 3 步预热），
再 3 步冻结几何外观训练。各阶段记录的总损失与原实现相同，固定几何和相机状态逐位未变。
训练权重不逐位一致：最大差约 0.002；未修改的原工程同种子重跑也出现约 0.002 的差异。
这与 CUDA 反向的非确定性一致，详细最大差和 RMS 在 JSON 中并列报告，不宣称训练权重逐位复现。

论文表格和图像均在新目录成功再生成，且与原文件逐字节一致；论文源文件和已有 PDF 未改动。

真实 Cat 最终权重的命令行 test 单帧渲染与 VGG LPIPS 评价也已完成。
18 份最终 checkpoint 均与来源逐字节一致，且为独立文件；没有对权重执行转换或重新保存。

## 公开项目结构与依赖验证

代码改为 lisags 包并增加三个控制台入口；模型、几何、渲染和训练计算经 AST 对照，
除导入路径外未变。已在临时隔离环境安装本项目及 gsplat submodule，并重新编译 CUDA。

发现原环境使用 GLM 0.9.9.9，而 gsplat v1.5.3 默认嵌套 GLM 1.0.2。
默认版本的 Cat 第一帧最大 RGB 差为 0.0006702；JIT 和 setup.py 构建均复现。
固定原 GLM 提交后，18 场景各一帧原分辨率 RGB/alpha 与原环境逐位一致。
安装脚本记录了这个依赖覆盖，详见 [3DGS 后端](../../third_party/README.md)。

新的 lisags-run 入口已用真实 Cat 四帧训练/两帧测试跑通短流程：4 步几何/联合训练、
2 步精修、完整小样本 test 渲染及 LPIPS。临时产物已清理，记录在
[package_validation.json](package_validation.json)。验证复用现有 PyTorch/科学计算库，
未从零求解新的 Conda 环境，也未重跑完整 38k 训练。
