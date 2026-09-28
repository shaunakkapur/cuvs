#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_diskann)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  set(diskann_override "${patch_dir}/diskann_override.json")
  if(CUVS_DISKANN_READER)
    # The optional CPU reader (c/diskann) reuses this dependency. Its patches are
    # applied only for that build, so benchmark behavior is unchanged. Pin the
    # commit the patches were written against instead of the moving cpp_main.
    file(READ "${diskann_override}" reader_override)
    string(REPLACE "\${current_json_dir}" "${patch_dir}" reader_override "${reader_override}")
    string(JSON reader_override SET "${reader_override}" packages diskann git_tag
           "\"78256bbab4685e1774e78d331e081a153be26823\"")
    set(reader_patches
        "diskann-bounded-pq.diff|Add a PQ encoding block-size parameter so preparation memory is bounded"
        "diskann-result-count-check.diff|Reject searches that return fewer than k results instead of reading past the result buffer"
        "diskann-safe-linux-aio.diff|Replace exit() on Linux AIO errors with synchronous reads that throw")
    foreach(entry IN LISTS reader_patches)
      string(REPLACE "|" ";" entry "${entry}")
      list(GET entry 0 patch_file)
      list(GET entry 1 patch_issue)
      string(JSON patch_count LENGTH "${reader_override}" packages diskann patches)
      string(JSON reader_override SET "${reader_override}" packages diskann patches ${patch_count}
             "{\"file\":\"${patch_dir}/${patch_file}\",\"issue\":\"${patch_issue}\",\"fixed_in\":\"\"}")
    endforeach()
    set(diskann_override "${CMAKE_CURRENT_BINARY_DIR}/diskann-reader-override.json")
    file(WRITE "${diskann_override}" "${reader_override}")
  endif()
  rapids_cpm_package_override("${diskann_override}")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(diskann
    VERSION_VAR version
    FIND_VAR find_args
    CPM_VAR cpm_args
  )

  rapids_cpm_find(diskann ${version} ${find_args}
          GLOBAL_TARGETS diskann
          CPM_ARGS ${cpm_args}
          OPTIONS
          "PYBIND OFF"
          "UNIT_TEST OFF"
          "RESTAPI OFF"
          "PORTABLE OFF")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(diskann)

  if(NOT TARGET diskann::diskann)
      target_include_directories(diskann INTERFACE "$<BUILD_INTERFACE:${diskann_SOURCE_DIR}/include>")
      add_library(diskann::diskann ALIAS diskann)
  endif()
endfunction()
find_and_configure_diskann()
