/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.DiskAnnIndex;
import com.nvidia.cuvs.VamanaIndex;
import com.nvidia.cuvs.VamanaIndexParams;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * Builds one DiskANN disk index for the vectors of a merged segment: cuVS builds a Vamana graph on
 * the GPU, then the DiskANN C API writes the disk layout and PQ files. Graph ordinal i is the i-th
 * vector in merge order, which is the flat format's ordinal i.
 */
final class DiskAnnGraphBuilder {
  static final int MIN_ROWS = 256; // PQ trains 256 centroids.
  private static final int GRAPH_DEGREE = 32;

  private DiskAnnGraphBuilder() {}

  /**
   * Writes the vectors to {@code data} in DiskANN's matrix format.
   *
   * @return the number of vectors, or -1 (with a partial file) once more than {@code maxRows} are
   *     seen
   */
  static int spool(FloatVectorValues values, Path data, int maxRows) throws IOException {
    int dimension = values.dimension(), count = 0;
    var row = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    try (var out =
        FileChannel.open(data, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      out.position(2 * Integer.BYTES);
      var iterator = values.iterator();
      while (iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        if (++count > maxRows) return -1;
        row.clear();
        row.asFloatBuffer().put(values.vectorValue(iterator.index()));
        out.write(row);
      }
      var header = ByteBuffer.allocate(2 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      out.write(header.putInt(count).putInt(dimension).flip(), 0);
    }
    return count;
  }

  /** Builds the graph for {@code data} and writes {@link DiskAnnIndex#PREPARED_FILES}. */
  static void build(Path data, int count, int dimension, Path prefix, int pqChunks)
      throws IOException {
    Path graph = prefix.resolveSibling(prefix.getFileName() + ".vamana");
    buildGraph(data, graph, count, dimension);
    // PQ training reads at most 256,000 rows; encoding reads that many at a time.
    int rows = Math.min(count, 256_000);
    DiskAnnIndex.prepare(data, graph, prefix, pqChunks, rows, rows);
  }

  private static void buildGraph(Path data, Path graph, int count, int dimension)
      throws IOException {
    var params =
        new VamanaIndexParams.Builder()
            .withGraphDegree(GRAPH_DEGREE)
            .withVisitedSize(2 * GRAPH_DEGREE)
            .withMetric(VamanaIndexParams.CuvsDistanceType.L2Expanded)
            .build();
    var matrix = CuVSMatrix.hostBuilder(count, dimension, CuVSMatrix.DataType.FLOAT);
    var row = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    try (var in = FileChannel.open(data, StandardOpenOption.READ)) {
      in.position(2 * Integer.BYTES);
      float[] vector = new float[dimension];
      for (int i = 0; i < count; i++) {
        row.clear();
        while (row.hasRemaining()) if (in.read(row) < 0) throw new IOException("Truncated spool");
        row.flip().asFloatBuffer().get(vector);
        matrix.addVector(vector);
      }
    }
    try (var dataset = matrix.build();
        var resources = CuVSResources.create();
        var index =
            VamanaIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(params)
                .build()) {
      index.serialize(graph, false);
    } catch (Throwable e) {
      if (e instanceof Error error && !(error instanceof LinkageError)) throw error;
      throw new IOException("cuVS Vamana graph build failed", e);
    }
  }
}
