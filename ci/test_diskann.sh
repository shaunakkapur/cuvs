#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Builds the optional DiskANN C library and runs its CPU tests. See c/diskann/README.md.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
: "${CONDA_PREFIX:?Activate a cuVS development environment}"
: "${JAVA_HOME:?Set JAVA_HOME to JDK 22}"
cd "$repo_dir"
diskann_build="${DISKANN_BUILD:-$repo_dir/c/diskann/build}"
maven_repo="${MAVEN_LOCAL_REPO:-$diskann_build/m2}"
bash c/diskann/build.sh
cmake -S java/internal -B java/internal/build -DFIND_CUVS_CPP=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake -E copy_directory c/include/cuvs java/internal/build/bindings/include/cuvs
(cd java && bash panama-bindings/generate-bindings.sh)
mvn -B -f java/cuvs-java/pom.xml -Pdiskann -Dit.test=DiskAnnIndexIT \
  -Dmaven.repo.local="$maven_repo" -Ddiskann.library="$diskann_build/libcuvs_diskann.so" \
  -Dmaven.javadoc.skip=true install
