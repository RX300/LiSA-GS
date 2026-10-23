# LisaGS 本轮重跑与 baseline 对比

核查日期：2026-10-07T04:31:31+09:00。

**18/18 场景、5691 张官方 test 全部完成。** Pikachu 先前的精修 OOM 已在空闲 GPU1 补跑完成，保留其原 30k 几何权重和失败记录。
所有新指标均来自独立项目本轮实际训练，没有用原工程结果补缺。

场景等权平均；PSNR/SSIM 越高越好，LPIPS 越低越好。Baseline 采用已有共同 GT 重评价，本轮采用同定义的原生评价。
全体 5691 张 GT 的字节、frame_index/name、测试帧数已核对，见 [审计](comparison_audit.json)。新指标的逐帧均值也已复核。

## all（18 场景）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| LisaGS (new) | 33.4681 | 0.95195 | 0.05641 |
| LiSA-staged (PORT-GS) | 33.4722 | 0.95199 | 0.05636 |
| GS3 | 31.4908 | 0.93919 | 0.06898 |
| SSD-GS | 31.5857 | 0.93878 | 0.06847 |
| OLAT-Gaussians | 30.9547 | 0.93303 | 0.07549 |
| RNG | 25.8283 | 0.89317 | 0.13062 |

## Real_NRHints（7 场景）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| LisaGS (new) | 29.2667 | 0.90930 | 0.10068 |
| LiSA-staged (PORT-GS) | 29.2779 | 0.90938 | 0.10050 |
| GS3 | 27.2790 | 0.88446 | 0.11564 |
| SSD-GS | 27.2008 | 0.88239 | 0.11712 |
| OLAT-Gaussians | 27.2569 | 0.88614 | 0.11437 |
| RNG | 20.9416 | 0.82795 | 0.20324 |

## Synthetic_GS3（6 场景）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| LisaGS (new) | 32.3900 | 0.97038 | 0.03821 |
| LiSA-staged (PORT-GS) | 32.4001 | 0.97042 | 0.03823 |
| GS3 | 31.9732 | 0.96669 | 0.04896 |
| SSD-GS | 31.5307 | 0.96503 | 0.04965 |
| OLAT-Gaussians | 27.1130 | 0.93993 | 0.07931 |
| RNG | 22.1399 | 0.89477 | 0.13178 |

## Synthetic_SSS-GS（5 场景）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| LisaGS (new) | 40.6439 | 0.98953 | 0.01628 |
| LiSA-staged (PORT-GS) | 40.6308 | 0.98951 | 0.01633 |
| GS3 | 36.8084 | 0.98283 | 0.02768 |
| SSD-GS | 37.7907 | 0.98623 | 0.02292 |
| OLAT-Gaussians | 40.7417 | 0.99039 | 0.01647 |
| RNG | 37.0956 | 0.98258 | 0.02756 |

## 逐场景 PSNR

| 数据族 / 场景 | LisaGS 本轮 | PORT-GS 最终版 | GS³ | SSD-GS | OLAT-GS | RNG | 对最佳 baseline 的差值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real_NRHints / Cat | 25.277 | 25.329 | 19.124 | 18.088 | 22.431 | 20.988 | +2.846 |
| Real_NRHints / CatSmall | 33.026 | 33.017 | 33.283 | 33.829 | 31.579 | 18.737 | -0.803 |
| Real_NRHints / CupFabric | 35.239 | 35.307 | 35.164 | 35.983 | 36.404 | 21.411 | -1.165 |
| Real_NRHints / Fish | 28.967 | 28.967 | 22.410 | 23.268 | 23.848 | 24.111 | +4.856 |
| Real_NRHints / FurScene | 24.482 | 24.474 | 26.093 | 25.151 | 24.769 | 24.120 | -1.611 |
| Real_NRHints / Pikachu | 31.656 | 31.649 | 31.155 | 30.694 | 31.064 | 17.271 | +0.500 |
| Real_NRHints / Pixiu | 26.220 | 26.203 | 23.724 | 23.391 | 20.703 | 19.953 | +2.496 |
| Synthetic_GS3 / AnisoMetal | 27.196 | 27.148 | 27.513 | 28.001 | 27.506 | 24.316 | -0.804 |
| Synthetic_GS3 / Drums | 30.449 | 30.451 | 31.517 | 31.541 | 20.639 | 15.442 | -1.092 |
| Synthetic_GS3 / FurBall | 36.011 | 36.113 | 35.832 | 34.996 | 35.564 | 17.609 | +0.179 |
| Synthetic_GS3 / Hotdog | 35.196 | 35.219 | 33.315 | 32.944 | 29.675 | 29.397 | +1.880 |
| Synthetic_GS3 / Lego | 30.829 | 30.801 | 30.906 | 29.717 | 20.266 | 16.784 | -0.077 |
| Synthetic_GS3 / Translucent | 34.658 | 34.668 | 32.756 | 31.986 | 29.028 | 29.292 | +1.902 |
| Synthetic_SSS-GS / bunny_small | 39.919 | 39.857 | 34.416 | 38.486 | 40.427 | 37.852 | -0.508 |
| Synthetic_SSS-GS / candle_small | 45.093 | 45.196 | 38.743 | 38.737 | 45.081 | 41.883 | +0.013 |
| Synthetic_SSS-GS / dragon_small | 37.363 | 37.378 | 36.104 | 37.377 | 37.788 | 35.235 | -0.425 |
| Synthetic_SSS-GS / soap_small | 41.046 | 40.906 | 38.347 | 38.365 | 41.420 | 40.280 | -0.374 |
| Synthetic_SSS-GS / statue_small | 39.799 | 39.817 | 36.432 | 35.988 | 38.993 | 30.228 | +0.806 |

完整三项逐场景指标：[CSV](baseline_comparison.csv)；均值：[CSV](baseline_comparison_summary.csv)；[对比图](../figures/baseline_comparison.png)。

## 结论与边界

- 与原 PORT-GS 最终版的平均 PSNR 差 -0.0041 dB；这是单次同种子复跑，不构成种子方差估计。
- 对各 baseline 的逐场景 PSNR 胜场（共 18）：{'GS3': 13, 'SSD-GS': 12, 'OLAT-Gaussians': 12, 'RNG': 18}。
- SSS 数据族 PSNR、SSIM 均值仍低于 OLAT-GS，LPIPS 略优；不能宣称每类、每场景都领先。
- 相对各场景最佳 baseline，FurScene、CupFabric、Drums 分别落后约 1.61、1.17、1.09 dB。
- 这是本地复现比较；训练预算、训练标定方式与随机性并非完全相同，不等同于等预算比较或文献 SOTA。SSS-GS 方法不纳入，SSS-GS 数据集纳入。

## 来源

- 本轮：`runs/full_dataset_20261006/<family>/<scene>/appearance/test/metrics.json`。
- Baseline 与原 LiSA：`runs/lisa_supplementary_experiments/common_ground_truth/<family>/<scene>/<method>.json`。
- Pikachu 失败记录：`appearance_failed_oom.log`；恢复记录：批次目录的 `pikachu_recovery.log`。
