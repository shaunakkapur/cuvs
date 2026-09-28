#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
    # One dependency/pin, with reader-only patches. Do not change benchmark I/O behavior.
    file(READ "${diskann_override}" reader_override)
    string(REPLACE "\${current_json_dir}" "${patch_dir}" reader_override "${reader_override}")
    file(GLOB reader_patches "${patch_dir}/diskann-*.diff")
    foreach(patch IN LISTS reader_patches)
      string(JSON patch_count LENGTH "${reader_override}" packages diskann patches)
      string(JSON reader_override SET "${reader_override}" packages diskann patches ${patch_count}
        "{\"file\":\"${patch}\",\"issue\":\"Experimental CPU reader safety and bounded buffers\",\"fixed_in\":\"\"}")
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
