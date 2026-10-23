# 传输与渲染

- `lisags/models/base.py`：最终方法需要的方向 Fourier 编码和 `TransportBase`。
- `lisags/models/light_atlas.py`：SVBRDF、学习法线、通量编码、深度矩、图集金字塔、可见性、传输与高光。
- `lisags/models/__init__.py`：唯一方法构造与 CLI 字段，无多方法注册器或兼容桥。
- `lisags/rendering/renderer.py`：3DGS 光源投影、协方差法线、逐高斯着色和像素着色；保留论文的 per-Gaussian shadow 消融。
- `lisags/geometry/gaussians.py`：高斯参数、Adam 参数组、相机包围盒与尺度/opacity 投影。
- `lisags/geometry/refinement.py`：继承原 gsplat 增密策略，按梯度和点数预算选择复制/分裂，保留 opacity 重置次序。

相机图像和光空间图集都使用 gsplat 1.5.3；其原始 CUDA 后端通过 `third_party/gsplat` submodule 固定，安装后由包导入。
`shading=gaussian` 用于联合训练，`shading=pixel` 用于最终精修和推理；两者属于同一个最终方法。
