#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
module_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
: "${CONDA_PREFIX:?Activate a cuVS environment with MKL, Boost, libaio and gperftools development packages}"
graph_build="${DISKANN_BUILD:-$module_dir/build}"
cmake -S "$module_dir" -B "$graph_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX${DISKANN_DEPS:+;$DISKANN_DEPS}" "$@"
cmake --build "$graph_build" --target cuvs_diskann --parallel 2
