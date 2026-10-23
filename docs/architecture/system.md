# System architecture

All project code lives in the `lisags` Python package. `pyproject.toml` exposes
three entry points without wrapper scripts:

| Command | Implementation | Responsibility |
| --- | --- | --- |
| `lisags-run` | `lisags/cli/run_staged.py` | Final two-stage recipe and test evaluation |
| `lisags-train` | `lisags/training/trainer.py` | Single-stage training loop |
| `lisags-evaluate` | `lisags/evaluation/evaluator.py` | Checkpoint rendering and metrics |

The training loop keeps the original order: sample → calibration → zero gradients
→ learning-rate schedule → render → loss → backward → optimizers → densification
→ projection → checkpoint. Moving files does not alter these operations.

`data/` decodes images and metadata. `geometry/` owns Gaussian parameters, cameras
and refinement. `rendering/` prepares camera/light rasterizations and receivers.
`models/` evaluates LiSA's light-space atlas and neural appearance.
`evaluation/` owns the observation encoding and scores, outside linear shading.

The [gsplat submodule](../../third_party/README.md) provides the differentiable
3DGS CUDA backend. Project Gaussian state and upstream CUDA rasterization are
separate parts of the same 3DGS pipeline. The package imports the installed,
pinned gsplat; the submodule must be installed as described in the README.

Checkpoints retain the original `light_atlas` representation and tensor keys;
no model-state conversion is required.
