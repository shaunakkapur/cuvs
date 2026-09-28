# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
cmake_minimum_required(VERSION 4.0)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(../../cmake/rapids_config.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-find)
  include(rapids-export)
  project(cuvs_diskann LANGUAGES C CXX)
  rapids_cpm_init()
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"
   OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(
    FATAL_ERROR
      "The experimental DiskANN reader currently requires Linux x86_64")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(BUILD_SHARED_LIBS OFF)
set(CUVS_DISKANN_READER ON)
find_package(MKL CONFIG REQUIRED)
include(../../cpp/cmake/thirdparty/get_diskann.cmake)
# CPM reports failed patches without stopping. Never build an unsafe reader in
# that case.
find_package(Git REQUIRED)
file(GLOB diskann_reader_patches
     "${CMAKE_CURRENT_LIST_DIR}/../../cpp/cmake/patches/diskann-*.diff")
foreach(patch IN LISTS diskann_reader_patches)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${diskann_SOURCE_DIR}" apply --reverse
            --check "${patch}"
    RESULT_VARIABLE applied
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT applied EQUAL 0)
    message(
      FATAL_ERROR
        "Required DiskANN patch is missing: ${patch}; recreate the dependency cache"
    )
  endif()
endforeach()
find_package(OpenMP REQUIRED)
find_path(AIO_INCLUDE libaio.h REQUIRED)
find_library(AIO_LIBRARY aio REQUIRED)
find_library(TCMALLOC_LIBRARY tcmalloc REQUIRED)
if(TARGET cuvs_c)
  set(CUVS_C_LIBRARY cuvs_c)
else()
  find_library(CUVS_C_LIBRARY cuvs_c REQUIRED)
endif()
find_path(CUVS_C_INCLUDE cuvs/core/export.h REQUIRED)
find_path(CUDA_RUNTIME_INCLUDE cuda_runtime.h
          PATH_SUFFIXES targets/x86_64-linux/include REQUIRED)
find_path(DLPACK_INCLUDE dlpack/dlpack.h REQUIRED)
add_library(cuvs_diskann SHARED ../src/neighbors/diskann.cpp)
target_compile_features(cuvs_diskann PRIVATE cxx_std_20)
target_compile_definitions(cuvs_diskann PRIVATE USE_AVX2 MKL_ILP64)
target_compile_options(cuvs_diskann PRIVATE -mavx2 -mfma)
target_include_directories(
  cuvs_diskann PRIVATE ../include "${CUVS_C_INCLUDE}" "${CUDA_RUNTIME_INCLUDE}"
                       "${DLPACK_INCLUDE}" "${AIO_INCLUDE}")
target_link_libraries(
  cuvs_diskann PRIVATE diskann::diskann "${CUVS_C_LIBRARY}" OpenMP::OpenMP_CXX
                       MKL::MKL "${AIO_LIBRARY}" "${TCMALLOC_LIBRARY}")
target_link_directories(cuvs_diskann PRIVATE "${MKL_ROOT}/lib")
target_link_options(cuvs_diskann PRIVATE -Wl,--no-undefined)
install(TARGETS cuvs_diskann LIBRARY DESTINATION lib)
