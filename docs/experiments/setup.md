# Setup and experiment protocol

## A new machine

Follow the [root README](../../README.md#environment-setup). The tested stack is Linux,
Python 3.10, PyTorch 2.4.1 / torchvision 0.19.1, CUDA toolkit 12.1 and gsplat 1.5.3.
The NVIDIA driver must support the CUDA runtime. A working GCC/G++ compiler and
`nvcc` are required for gsplat's JIT build, even though PyTorch includes CUDA runtime libraries.

The PyTorch command follows the [official 2.4.1 installation matrix](https://docs.pytorch.org/get-started/previous-versions/#v241).
The `BUILD_NO_CUDA=1` installation mode is provided by gsplat's own setup.py and
defers compilation to the first rasterization. Use `scripts/install_backend.sh` to retain the experiment's GLM 0.9.9.9 pin.
Use the pinned submodule rather
than installing an unpinned PyPI/latest version. Inspect it with:

```bash
git submodule status --recursive
nvcc --version
python -c "import torch, gsplat; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), gsplat.__version__)"
lisags-train --help
lisags-evaluate --help
```

Keep `CUDA_HOME=$CONDA_PREFIX` when using the Conda CUDA toolkit from the README.
If you use an existing system CUDA 12.1 toolkit instead, set `CUDA_HOME` to its
installation root. On servers without a display, OpenCV may need the system
`libGL.so.1` runtime (Ubuntu package `libgl1`).

## Canonical recipe

`configs/staged.json` contains the final 30k+8k training parameters and family
overrides. Run from the repository root after an editable installation:

```bash
CUDA_VISIBLE_DEVICES=0 lisags-run --data-root /path/to/data \
  --family Synthetic_GS3 --scene Lego --output runs/lego
```

The first 8k steps bootstrap geometry and transition the radiometric target;
the following 22k jointly optimize per-Gaussian shading. The next 8k use pixel
shading with frozen geometry and camera calibration. Evaluation covers official
test frames with their original calibration. Seed 0, no test checkpoint selection.

The runner records recipe.json, commands.json and source.tar. Each stage writes
config.json, split.json, history.jsonl and last.pt. Checkpoints are model states,
not complete optimizer-resume snapshots. CUDA backward can be nondeterministic;
see [validation and results](results.md).

## Backend pin

The nested GLM revision intentionally differs from gsplat's default gitlink.
The installer is the reproducible source of this override; see [backend pins](../../third_party/README.md).
