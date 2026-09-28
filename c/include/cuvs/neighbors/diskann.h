/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuvs/core/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup diskann_c_reader Experimental DiskANN CPU reader
 * @{
 */

/**
 * @brief Handle for trusted float32, squared-L2 DiskANN artifacts.
 *
 * Calls on a handle must be serialized. Close after all searches complete.
 * This optional backend is not a parser for untrusted indexes.
 */
typedef uintptr_t cuvsDiskAnnIndex_t;

/**
 * @brief Open an index with 256..1M rows and 1..128 dimensions.
 * @param prefix Path prefix shared by disk-layout and PQ files.
 * @param dimension Expected vector dimension.
 * @param count Expected number of vectors.
 * @param handle Receives the owned handle; close it with cuvsDiskAnnClose.
 * @return CUVS_SUCCESS on success, otherwise CUVS_ERROR with cuvsGetLastErrorText.
 */
CUVS_EXPORT cuvsError_t cuvsDiskAnnOpen(const char* prefix,
                                        uint32_t dimension,
                                        uint32_t count,
                                        cuvsDiskAnnIndex_t* handle);

/**
 * @brief Search a loaded index using squared L2 distance.
 * @param handle An open reader handle.
 * @param query Finite float32 query with dimension elements.
 * @param dimension Dimension used when opening the reader.
 * @param k Number of requested results, between 1 and the index size.
 * @param search_list Search list size, between k and the index size.
 * @param ids Output buffer of k vector ordinals.
 * @param distances Output buffer of k squared distances.
 * @return CUVS_SUCCESS on success, otherwise CUVS_ERROR; discard outputs on error.
 */
CUVS_EXPORT cuvsError_t cuvsDiskAnnSearch(cuvsDiskAnnIndex_t handle,
                                          const float* query,
                                          uint32_t dimension,
                                          uint32_t k,
                                          uint32_t search_list,
                                          uint64_t* ids,
                                          float* distances);

/**
 * @brief Release an owned handle. Zero is accepted; a closed handle must not be reused.
 * @param handle Reader handle to release.
 * @return CUVS_SUCCESS on success, otherwise CUVS_ERROR.
 */
CUVS_EXPORT cuvsError_t cuvsDiskAnnClose(cuvsDiskAnnIndex_t handle);

/**
 * @brief Convert a cuVS Vamana graph to disk layout and plain PQ.
 * @param data Little-endian float32 matrix with uint32 row and dimension headers.
 * @param graph Serialized cuVS Vamana graph for the same vector order.
 * @param prefix New output prefix; its artifact files must not already exist.
 * @param chunks PQ chunks, between 1 and the vector dimension.
 * @param sample_rows Maximum PQ training rows, between 256 and 256000.
 * @param block_rows Encoding block rows, between 1 and 256000.
 * @return CUVS_SUCCESS on success, otherwise CUVS_ERROR with cuvsGetLastErrorText.
 */
CUVS_EXPORT cuvsError_t cuvsDiskAnnPrepare(const char* data,
                                           const char* graph,
                                           const char* prefix,
                                           uint32_t chunks,
                                           uint32_t sample_rows,
                                           uint32_t block_rows);

/** @} */

#ifdef __cplusplus
}
#endif
