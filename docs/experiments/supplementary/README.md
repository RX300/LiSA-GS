# LiSA 补充实验

本目录是本轮实验的唯一阅读入口。模型固定为 LiSA-staged；排除 SSS-GS **方法**，保留 Synthetic_SSS-GS **数据**。

本轮已全部验收：65项训练/精修、9项测量、18场景复评。

## 先看这些文件

- [results.md](results.md)：结果概览、各组结论及耗时/速度对照。
- [verification.json](verification.json)：最终核验记录。
- [progress.json](progress.json)：65 项训练/精修任务的完成、运行、等待与失败数量，每小时及队列结束时更新。
- [results.csv](results.csv)：逐场景、逐配置、逐 seed 指标及原始结果路径；`reused` 标明复用结果。
- [summary.csv](summary.csv)：同组、同配置、同 seed 的场景等权平均；组未完成时均值留空。
- `render_efficiency.csv`：9项测量已完成，记录固定/变化灯位的FPS、显存和模型大小。
- `repeatability.csv`、`paired_deltas.csv`：三seed统计已齐全，包含逐场景均值/标准差，以及完整模型相对无传输、局部残差的配对差值。
- [protocol.md](protocol.md)：固定实验矩阵、对照含义、训练与评价口径。

## 迁移范围

这是 PORT-GS 已完成补充实验的数表快照，不是新项目中的活动队列。
本项目复制了 common_ground_truth 的全部报告和论文用到的少量消融对照图；
65 项历史训练的非最终权重、源码归档和旧调度器仍留在原工程。
结果 CSV 中的绝对路径是历史来源记录。

新项目的最终训练入口为 `lisags-run`，配方为 `configs/staged.json`，详见
[运行协议](../setup.md)。论文图表直接读取这里的数表，无需重新运行旧消融队列。
