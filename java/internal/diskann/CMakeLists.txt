# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cuvs_diskann LANGUAGES CXX)
endif()

# Optional, CPU-only integration. Reuse a separately prepared, pinned PIC DiskANN
# archive; never download or patch a caller's dependency tree during compilation.
set(DISKANN_SOURCE "" CACHE PATH "Patched Microsoft DiskANN checkout")
set(DISKANN_ARCHIVE "" CACHE FILEPATH "PIC libdiskann.a from that checkout")
set(DISKANN_DEPS "" CACHE PATH "Environment containing DiskANN headers/libraries")
foreach(required DISKANN_SOURCE DISKANN_ARCHIVE DISKANN_DEPS)
  if(NOT EXISTS "${${required}}")
    message(FATAL_ERROR "Set ${required}; see README.md")
  endif()
endforeach()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${DISKANN_SOURCE}" rev-parse HEAD
                OUTPUT_VARIABLE diskann_revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT diskann_revision STREQUAL "78256bbab4685e1774e78d331e081a153be26823")
  message(FATAL_ERROR "DiskANN revision differs from the tested pin")
endif()
file(GLOB patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
foreach(patch IN LISTS patches)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${DISKANN_SOURCE}" apply --reverse --check "${patch}"
                  RESULT_VARIABLE patch_status ERROR_VARIABLE patch_error)
  if(NOT patch_status EQUAL 0)
    message(FATAL_ERROR "Required patch not applied: ${patch}\n${patch_error}")
  endif()
endforeach()

find_package(JNI REQUIRED COMPONENTS JVM)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
add_library(cuvs_diskann SHARED diskann_jni.cpp)
target_compile_features(cuvs_diskann PRIVATE cxx_std_17)
target_compile_definitions(cuvs_diskann PRIVATE USE_AVX2 MKL_ILP64)
target_compile_options(cuvs_diskann PRIVATE -mavx2 -mfma)
target_include_directories(cuvs_diskann PRIVATE ${JNI_INCLUDE_DIRS} "${DISKANN_SOURCE}/include" "${DISKANN_DEPS}/include")
target_link_directories(cuvs_diskann PRIVATE "${DISKANN_DEPS}/lib")
target_link_libraries(cuvs_diskann PRIVATE "${DISKANN_ARCHIVE}" OpenMP::OpenMP_CXX
  aio tcmalloc mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 Threads::Threads m ${CMAKE_DL_LIBS})
target_link_options(cuvs_diskann PRIVATE -Wl,--no-undefined)
set_target_properties(cuvs_diskann PROPERTIES BUILD_RPATH "${DISKANN_DEPS}/lib" INSTALL_RPATH "$ORIGIN")
install(TARGETS cuvs_diskann LIBRARY DESTINATION lib)
