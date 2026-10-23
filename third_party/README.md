# External 3DGS backend

[gsplat](https://github.com/nerfstudio-project/gsplat) is a Git submodule at
`gsplat/`, pinned to **v1.5.3**, commit
`937e29912570c372bed6747a5c9bf85fed877bae`.
The experiments used **GLM 0.9.9.9**, commit
`5c46b9c07008ae65cb81ab79cd677ecc1934b903`, rather than gsplat's default GLM 1.0.2.
The backend installer explicitly checks out this existing upstream commit.

```bash
bash scripts/install_backend.sh
```

Run these commands from the LisaGS repository root after installing PyTorch and
the CUDA toolkit as described in the root README. `BUILD_NO_CUDA=1` defers CUDA
compilation to the first render; it does not disable GPU rendering.
Do not replace either pin when reproducing results. A plain recursive submodule
update resets GLM to gsplat's default; rerun the installer before rebuilding.
Git will show gsplat as modified because its nested GLM checkout intentionally
differs from the upstream gitlink. There are no edits to gsplat or GLM source files.

## What implements 3DGS?

| Responsibility | Source |
| --- | --- |
| Trainable means, scales, rotations, opacities and material features | `lisags/geometry/gaussians.py` |
| Densification, pruning and opacity-reset schedule | `lisags/geometry/refinement.py`, gsplat `strategy/` |
| Camera/light passes, receiver reconstruction and composition | `lisags/rendering/renderer.py` |
| Differentiable projection, tile sorting and rasterization | `gsplat/gsplat/rendering.py`, `gsplat/gsplat/cuda/` |
| LiSA light-space atlas, neural material and transport | `lisags/models/light_atlas.py` |

The CUDA backend is gsplat, not the Graphdeco `diff-gaussian-rasterization`
extension. Switching backends would be a method change and is not part of this
reorganization. The 67 upstream Python/CUDA/C++ source files (excluding GLM)
match the existing experimental environment byte for byte. All 428 GLM files
also match at the experimental pin. The default GLM 1.0.2 build changed the first
Cat test render by up to 0.0006702, so using the default would not reproduce the
original backend numerically.

Upstream license: [Apache-2.0](gsplat/LICENSE), retained within the submodule.
