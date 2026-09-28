#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare the fork's optional reader and Java artifacts. No GPU tests or libcuvs rebuild.
set -euo pipefail
module_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(cd -- "$module_dir/../../.." && pwd)"
graph_cache="${GRAPH_CACHE:-$repo_dir/.cache/diskann}"
: "${JAVA_HOME:?Set JAVA_HOME to JDK 22}"
: "${CONDA_PREFIX:?Activate the cuVS development environment}"
diskann_source="${DISKANN_SOURCE:-$graph_cache/source}"
diskann_build="${DISKANN_BUILD:-$graph_cache/build}"
diskann_deps="${DISKANN_DEPS:-$graph_cache/deps}"
maven_repo="${GRAPH_MAVEN_REPO:-$graph_cache/repository}"
revision=78256bbab4685e1774e78d331e081a153be26823

if [[ ! -e "$diskann_source" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/microsoft/DiskANN.git "$diskann_source"
  git -C "$diskann_source" checkout --detach "$revision"
fi
if [[ ! -e "$diskann_source/.git" || "$(git -C "$diskann_source" rev-parse HEAD)" != "$revision" ]]; then
  printf 'Expected DiskANN at %s. Existing source was left untouched.\n' "$revision" >&2
  exit 1
fi
for patch in "$module_dir"/patches/*.patch; do
  if git -C "$diskann_source" apply --reverse --check "$patch" 2>/dev/null; then continue; fi
  git -C "$diskann_source" apply --check "$patch"
  git -C "$diskann_source" apply "$patch"
done
expected_changes="$(sed -n 's|^+++ b/||p' "$module_dir"/patches/*.patch | sort -u)"
if [[ "$(git -C "$diskann_source" diff --name-only | sort)" != "$expected_changes" ]]; then
  printf 'DiskANN has changes outside the recorded patches; stopped.\n' >&2
  exit 1
fi
if [[ ! -d "$diskann_deps" ]]; then
  CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$graph_cache/conda-pkgs}" \
    mamba env create -y -p "$diskann_deps" -f "$module_dir/environment.yml"
fi
"$diskann_deps/bin/cmake" -S "$diskann_source" -B "$diskann_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_PREFIX_PATH="$diskann_deps" -DCMAKE_CXX_FLAGS="-I$diskann_deps/include" \
  -DCMAKE_MAKE_PROGRAM="$diskann_deps/bin/ninja" \
  -DMKL_PATH="$diskann_deps/lib" -DMKL_INCLUDE_PATH="$diskann_deps/include" -DOMP_PATH="$diskann_deps/lib"
"$diskann_deps/bin/cmake" --build "$diskann_build" --target diskann --parallel 2
compiler="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$diskann_build/CMakeCache.txt")"
: "${compiler:?Missing DiskANN compiler}"
"$diskann_deps/bin/cmake" -S "$module_dir" -B "$module_dir/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$compiler" \
  -DDISKANN_SOURCE="$diskann_source" -DDISKANN_ARCHIVE="$diskann_build/src/libdiskann.a" -DDISKANN_DEPS="$diskann_deps"
"$diskann_deps/bin/cmake" --build "$module_dir/build" --parallel 2

# Reuse the existing header staging/generator and Maven projects. The current
# checkout's headers take precedence over headers from an installed older cuVS.
cmake -S "$repo_dir/java/internal" -B "$repo_dir/java/internal/build" \
  -DFIND_CUVS_CPP=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake -E copy_directory "$repo_dir/c/include/cuvs" "$repo_dir/java/internal/build/bindings/include/cuvs"
(cd "$repo_dir/java" && bash panama-bindings/generate-bindings.sh)
for module in cuvs-java; do
  mvn -B -Dmaven.repo.local="$maven_repo" -f "$repo_dir/java/$module/pom.xml" \
    -DskipTests -Dmaven.javadoc.skip=true install
done
printf 'Built %s and installed fork artifacts into %s\n' "$module_dir/build/libcuvs_diskann.so" "$maven_repo"
