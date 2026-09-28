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
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;

/** One controlled merge snapshot and one GPU-built graph with disk-backed ordinals. */
public final class DiskAnnGraphBuilder {
  public record Limits(int sampleRows, int maxGraphRows, long scratchBytes) {
    public Limits {
      if (sampleRows < 256
          || sampleRows > 256_000
          || maxGraphRows < 256
          || maxGraphRows > 4096
          || scratchBytes < 1024)
        throw new IllegalArgumentException("Invalid graph construction limits");
    }
  }

  public record Spool(Path data, Path mapping, int count, int dimension, float[][] sample) {}

  public record Built(
      Path prefix,
      String[] suffixes,
      int count,
      int subgraphs,
      int largestSubgraph,
      long matrixBytes,
      long scratchBytes) {}

  public static Spool spool(FloatVectorValues values, Path scratch, Limits limits)
      throws IOException {
    if (values.size() > limits.maxGraphRows() || values.dimension() > 128)
      throw new IOException("Segment exceeds configured single-graph row limit or 128 dimensions");
    int dimension = values.dimension(), count = 0;
    float[][] sample = new float[limits.sampleRows()][];
    var random = new SplittableRandom(731);
    Path data = scratch.resolve("vectors.bin"), mapping = scratch.resolve("graph.ordmap");
    try (var vectors = new RandomAccessFile(data.toFile(), "rw");
        var ids = new RandomAccessFile(mapping.toFile(), "rw")) {
      header(vectors, 0, dimension);
      header(ids, 0, 1);
      var iterator = values.iterator();
      int previous = -1;
      for (int doc = iterator.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = iterator.nextDoc()) {
        if (doc <= previous)
          throw new IOException("Merge document IDs are not strictly increasing");
        previous = doc;
        float[] vector = values.vectorValue(iterator.index());
        if (vector.length != dimension) throw new IOException("Vector dimension mismatch");
        for (float v : vector) {
          if (!Float.isFinite(v)) throw new IOException("Non-finite vector");
        }
        writeRow(vectors, vector);
        ids.writeInt(Integer.reverseBytes(doc));
        count++;
        int selected = count <= sample.length ? count - 1 : random.nextInt(count);
        if (selected < sample.length) sample[selected] = vector.clone();
        if (vectors.length() + ids.length() > limits.scratchBytes())
          throw new IOException("Spool scratch budget exceeded");
      }
      header(vectors, count, dimension);
      header(ids, count, 1);
    }
    if (count < 256)
      throw new IOException("Graph merge requires at least 256 vectors for PQ training");
    return new Spool(
        data, mapping, count, dimension, Arrays.copyOf(sample, Math.min(count, sample.length)));
  }

  public static Built build(Spool spool, Path scratch, Limits limits, int pqChunks)
      throws IOException {
    if (pqChunks == 0) pqChunks = Math.min(8, spool.dimension());
    if (pqChunks < 1 || pqChunks > spool.dimension())
      throw new IOException("Invalid PQ chunk count");
    // Conservative on-disk envelope, including native staging, sector padding and both
    // replicas of each vector. Reject before native construction, not after filling disk.
    long plannedScratch =
        Math.addExact(
            64L << 20,
            Math.multiplyExact(
                (long) spool.count(), 64L * spool.dimension() + 3L * 33 * 4 + 16384));
    if (plannedScratch > limits.scratchBytes())
      throw new IOException("Planned graph scratch exceeds budget");
    Path prefix = scratch.resolve("graph");
    if (spool.count() > limits.maxGraphRows())
      throw new IOException("Segment exceeds the single-graph row limit");
    Path graph = scratch.resolve("direct");
    buildGraph(spool.data(), graph, spool.count(), spool.dimension());
    checkScratch(scratch, limits);
    String[] nativeFiles =
        DiskAnnIndex.prepare(
            spool.data(), graph, prefix, pqChunks, limits.sampleRows(), pqBlockRows(limits));
    return validate(spool, scratch, limits, prefix, nativeFiles, 1, spool.count());
  }

  static int pqBlockRows(Limits limits) {
    // The encoder's block limit is independent of the GPU subgraph row budget.
    return Math.min(256_000, limits.maxGraphRows());
  }

  private static void buildGraph(Path data, Path graph, int count, int dimension)
      throws IOException {
    var params =
        new VamanaIndexParams.Builder()
            .withGraphDegree(32)
            .withVisitedSize(64)
            .withMetric(VamanaIndexParams.CuvsDistanceType.L2Expanded)
            .build();
    try {
      var matrixBuilder = CuVSMatrix.hostBuilder(count, dimension, CuVSMatrix.DataType.FLOAT);
      try (var matrix = matrixBuilder.build();
          var input = new RandomAccessFile(data.toFile(), "r")) {
        // A built host matrix owns its allocation; the builder fills that same matrix.
        input.seek(8);
        for (int i = 0; i < count; i++) matrixBuilder.addVector(row(input, dimension));
        try (var resources = CuVSResources.create();
            var index =
                VamanaIndex.newBuilder(resources)
                    .withDataset(matrix)
                    .withIndexParams(params)
                    .build()) {
          index.serialize(graph);
        }
      }
    } catch (Throwable e) {
      if (e instanceof Error error && !(error instanceof LinkageError)) throw error;
      throw new IOException("cuVS graph build failed", e);
    }
  }

  private static Built validate(
      Spool spool,
      Path scratch,
      Limits limits,
      Path prefix,
      String[] nativeFiles,
      int subgraphs,
      int largest)
      throws IOException {
    // Validate the private output before the codec copies any artifact into
    // the segment. Opening checks count/dimension; a probe checks ID/score use.
    try (var reader = DiskAnnIndex.open(prefix, spool.dimension(), spool.count());
        var dataFile = new RandomAccessFile(spool.data().toFile(), "r")) {
      float[] query = spool.sample()[0];
      var probe = reader.search(query, 1, Math.min(128, spool.count()));
      long ordinal = probe.ordinals()[0];
      float actual = probe.distances()[0];
      if (ordinal < 0 || ordinal >= spool.count() || !Float.isFinite(actual) || actual < 0)
        throw new IOException("Invalid private graph probe");
      dataFile.seek(8 + ordinal * spool.dimension() * Float.BYTES);
      double expected = squaredL2(query, row(dataFile, spool.dimension()));
      if (Math.abs(expected - actual) > Math.max(1e-4, expected * 1e-5))
        throw new IOException("Private graph probe does not match source vector");
    }
    String[] files = Arrays.copyOf(nativeFiles, nativeFiles.length + 1);
    files[nativeFiles.length] = ".ordmap";
    long bytes = checkScratch(scratch, limits);
    return new Built(
        prefix,
        files,
        spool.count(),
        subgraphs,
        largest,
        (long) largest * spool.dimension() * 4,
        bytes);
  }

  public static float[] row(RandomAccessFile input, int dimension) throws IOException {
    float[] row = new float[dimension];
    byte[] bytes = new byte[dimension * 4];
    input.readFully(bytes);
    java.nio.ByteBuffer.wrap(bytes)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()
        .get(row);
    return row;
  }

  static void writeRow(RandomAccessFile output, float[] row) throws IOException {
    byte[] bytes = new byte[row.length * 4];
    java.nio.ByteBuffer.wrap(bytes)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()
        .put(row);
    output.write(bytes);
  }

  static void header(RandomAccessFile file, int count, int dimension) throws IOException {
    file.seek(0);
    file.writeInt(Integer.reverseBytes(count));
    file.writeInt(Integer.reverseBytes(dimension));
  }

  static long checkScratch(Path scratch, Limits limits) throws IOException {
    long bytes = 0;
    try (var paths = Files.walk(scratch)) {
      for (Path path : paths.filter(Files::isRegularFile).toList()) bytes += Files.size(path);
    }
    if (bytes > limits.scratchBytes()) throw new IOException("Graph scratch budget exceeded");
    return bytes;
  }

  private static double squaredL2(float[] a, float[] b) {
    double total = 0;
    for (int i = 0; i < a.length; i++) {
      double delta = (double) a[i] - b[i];
      total += delta * delta;
    }
    return total;
  }
}
