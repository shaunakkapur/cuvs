/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Optional CPU DiskANN reader. Searches on one reader are serialized; distinct readers are
 * independent.
 */
public interface DiskAnnIndex extends Closeable {
  record Results(long[] ordinals, float[] distances) {}

  static DiskAnnIndex open(Path prefix, int dimension, int expectedCount) throws IOException {
    return CuVSProvider.provider().openDiskAnnIndex(prefix, dimension, expectedCount);
  }

  Results search(float[] query, int k, int searchList) throws IOException;

  /** Load the optional C library from an explicit path before opening readers. */
  static void loadLibrary(Path library) throws IOException {
    CuVSProvider.provider().loadDiskAnnLibrary(library);
  }

  static void prepare(Path data, Path graph, Path prefix, int pqChunks) throws IOException {
    prepare(data, graph, prefix, pqChunks, 4096, 4096);
  }

  /** Bounded plain-PQ conversion; returns the files produced relative to prefix. */
  static String[] prepare(
      Path data, Path graph, Path prefix, int pqChunks, int sampleRows, int blockRows)
      throws IOException {
    CuVSProvider.provider()
        .prepareDiskAnnIndex(data, graph, prefix, pqChunks, sampleRows, blockRows);
    return new String[] {
      "_disk.index", "_disk.index_medoids.bin", "_pq_pivots.bin", "_pq_compressed.bin"
    };
  }
}
