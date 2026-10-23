#!/usr/bin/env bash
# Reproduce the gsplat + GLM sources used by the final LiSA experiments.
set -eu
PROJECT_DIR=$(cd -- "$(dirname -- "$0")/.." && pwd)
git -C "$PROJECT_DIR" submodule update --init --recursive third_party/gsplat
# gsplat v1.5.3 defaults to GLM 1.0.2; the experiments used this 0.9.9.9 revision.
git -C "$PROJECT_DIR/third_party/gsplat/gsplat/cuda/csrc/third_party/glm" \
  checkout --detach 5c46b9c07008ae65cb81ab79cd677ecc1934b903
BUILD_NO_CUDA=1 python -m pip install --no-build-isolation "$PROJECT_DIR/third_party/gsplat"
