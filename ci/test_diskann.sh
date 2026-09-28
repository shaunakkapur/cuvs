#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional fork-CI/local entrypoint. Uses an existing cuVS + DiskANN development environment.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
: "${CONDA_PREFIX:?Activate the compatible cuVS development environment}"
: "${JAVA_HOME:?Use JDK 22}"
cd "$repo_dir"
bash c/diskann/build.sh
cmake -S java/internal -B java/internal/build -DFIND_CUVS_CPP=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake -E copy_directory c/include/cuvs java/internal/build/bindings/include/cuvs
(cd java && bash panama-bindings/generate-bindings.sh)
reader_library="${DISKANN_BUILD:-$repo_dir/c/diskann/build}/libcuvs_diskann.so"
mvn -B -f java/cuvs-java/pom.xml -Pdiskann -Dit.test=DiskAnnIndexIT \
  -Ddiskann.library="$reader_library" -Dmaven.javadoc.skip=true install
