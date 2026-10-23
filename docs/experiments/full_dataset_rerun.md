# LisaGS 三类数据集重新训练

**最终状态：18/18 场景、5691 张官方 test 全部完成。** [完整 baseline 对比](baseline_comparison.md)。

启动日期：2026-10-06（Asia/Tokyo）。本轮使用独立 LisaGS 包，复用 ssd-gs 环境。

- Synthetic_GS3：6 场景；Real_NRHints：7 场景；Synthetic_SSS-GS：5 场景。
- 共 18 场景，seed 0，全部官方 train，从头轮廓初始化。
- 每场景 30k 几何/联合训练 + 8k 固定几何外观精修，再渲染全部官方 test 并计算 PSNR/SSIM/LPIPS。
- 两张启动时空闲的物理 GPU1/3，每卡一个场景，跨数据族交错排队；不复用既有训练权重。
- 复用 lisags/cli/run_staged.py 的 --gpus 模式，不新建一次性实验脚本。

正式命令（项目根目录）：

```bash
python -m lisags.cli.run_staged --data-root <data_root> \
  --gpus 1 3 --output runs/full_dataset_20261006
```

输出：`runs/full_dataset_20261006/`；总日志：`runs/full_dataset_20261006.log`。
`status.json` 保存队列状态；每场景下 geometry.log / appearance.log / test.log 保存阶段日志。
`progress.jsonl` 每小时保存实际进程、GPU占用和训练进度，最终写 summary.json。
训练子进程由 tmux socket `lisags-full-dataset`、session `queue` 保持运行。

准确环境、启动命令和逐场景帧数见 [full_dataset_launch.json](full_dataset_launch.json)。
每批次与每场景都有 source.tar、recipe.json 和命令记录；原工程及迁移权重保持不变。

状态（2026-10-07）：17/18 场景完成。Pikachu 的 30k 几何阶段完成，精修因外部进程占用显存 OOM；
已保留失败日志，在空闲 GPU1 使用保存的命令补跑精修及评价。当前同场景对比见 [baseline_comparison.md](baseline_comparison.md)。

2026-10-07 最终核验：Pikachu 补跑精修及 200 帧 test 完成，本轮 18/18 场景共 5691 帧全部完成。
总结与 baseline 公平对比见 [baseline_comparison.md](baseline_comparison.md)。历史故障记录保留。
