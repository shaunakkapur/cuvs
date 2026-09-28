/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.Executors;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class DiskAnnIndexIT {
  @Rule public TemporaryFolder temporary = new TemporaryFolder();
  private Path directory;

  @org.junit.Before
  public void setUp() {
    directory = temporary.getRoot().toPath();
  }

  @org.junit.BeforeClass
  public static void loadLibrary() throws IOException {
    if (System.getProperty("diskann.library") == null)
      throw new IllegalStateException("Set diskann.library for the diskann profile");
    DiskAnnIndex.loadLibrary(Path.of(System.getProperty("diskann.library")));
  }

  @Test
  public void reportsNativeLibraryLinkFailuresAsIoErrors() {
    IOException failure =
        assertThrows(
            IOException.class,
            () -> DiskAnnIndex.loadLibrary(directory.resolve("missing-native-library.so")));
    assertTrue(failure.getCause() instanceof UnsatisfiedLinkError);
  }

  @Test
  public void rejectsMissingArtifactsAndInvalidDimensions() {
    Path prefix = directory.resolve("missing");
    assertThrows(IOException.class, () -> DiskAnnIndex.open(prefix, 16, 256));
    assertThrows(IllegalArgumentException.class, () -> DiskAnnIndex.open(prefix, 0, 256));
    assertThrows(IllegalArgumentException.class, () -> DiskAnnIndex.open(prefix, 16, 1_000_001));
  }

  @Test
  public void rejectsInvalidPreparationBeforeWritingOutputs() throws Exception {
    Path data = directory.resolve("vectors.bin"), prefix = directory.resolve("index");
    ByteBuffer matrix = ByteBuffer.allocate(8 + 256 * 4).order(ByteOrder.LITTLE_ENDIAN);
    matrix.putInt(256).putInt(1).putFloat(Float.NaN);
    Files.write(data, matrix.array());
    assertThrows(
        IOException.class, () -> DiskAnnIndex.prepare(data, directory.resolve("graph"), prefix, 1));
    assertThrows(
        IOException.class, () -> DiskAnnIndex.prepare(data, directory.resolve("graph"), prefix, 2));
    assertFalse(Files.exists(directory.resolve("index_disk.index")));
  }

  @Test
  public void rejectsMalformedArtifactsAndReportsTruncatedReads() throws Exception {
    Path prefix = prepareRing("read-failure", true);
    try (DiskAnnIndex reader = DiskAnnIndex.open(prefix, DIMENSION, COUNT)) {
      DiskAnnIndex.Results results = reader.search(new float[] {42, 8}, 5, COUNT);
      assertEquals(5, results.ordinals().length);
      assertEquals(5, results.distances().length);
    }
    int count = COUNT, dimension = DIMENSION;
    Path diskIndex = directory.resolve("read-failure_disk.index");
    overwriteLong(diskIndex, 8, count + 1L);
    IOException invalidDisk =
        assertThrows(IOException.class, () -> DiskAnnIndex.open(prefix, dimension, count));
    assertTrue(invalidDisk.getMessage().contains("Invalid disk index metadata"));
    overwriteLong(diskIndex, 8, count);

    Path compressed = directory.resolve("read-failure_pq_compressed.bin");
    overwriteInt(compressed, 0, 1_000_000);
    IOException invalidCompressed =
        assertThrows(IOException.class, () -> DiskAnnIndex.open(prefix, dimension, count));
    assertTrue(invalidCompressed.getMessage().contains("Invalid matrix file"));
    overwriteInt(compressed, 0, count);

    Path pivots = directory.resolve("read-failure_pq_pivots.bin");
    overwriteInt(pivots, 4096, 255);
    IOException invalidPivots =
        assertThrows(IOException.class, () -> DiskAnnIndex.open(prefix, dimension, count));
    assertTrue(invalidPivots.getMessage().contains("Invalid PQ pivot metadata"));
    overwriteInt(pivots, 4096, 256);

    try (DiskAnnIndex reader = DiskAnnIndex.open(prefix, dimension, count);
        FileChannel disk =
            FileChannel.open(
                directory.resolve("read-failure_disk.index"), StandardOpenOption.WRITE)) {
      disk.truncate(0);
      assertThrows(IOException.class, () -> reader.search(new float[] {42, 8}, 5, count));
    }
  }

  @Test
  public void reportsWhenTheGraphCannotProduceEnoughResults() throws Exception {
    Path prefix = prepareRing("disconnected", false);
    try (DiskAnnIndex reader = DiskAnnIndex.open(prefix, DIMENSION, COUNT)) {
      IOException failure =
          assertThrows(IOException.class, () -> reader.search(new float[] {42, 8}, 5, COUNT));
      assertTrue(failure.getMessage().contains("fewer results than requested"));
    }
  }

  @Test
  public void rejectsInvalidQueriesAndUseAfterClose() throws Exception {
    Path prefix = prepareRing("lifecycle", true);
    DiskAnnIndex reader = DiskAnnIndex.open(prefix, DIMENSION, COUNT);
    float[] query = {3, 4};
    assertThrows(IllegalArgumentException.class, () -> reader.search(new float[1], 5, 16));
    assertThrows(
        IllegalArgumentException.class, () -> reader.search(new float[] {Float.NaN, 0}, 5, 16));
    assertThrows(IllegalArgumentException.class, () -> reader.search(query, 5, 4));
    assertThrows(IllegalArgumentException.class, () -> reader.search(query, COUNT + 1, COUNT + 1));
    // Lucene may open a reader on one thread and search or close it on another.
    try (var executor = Executors.newSingleThreadExecutor()) {
      assertEquals(5, executor.submit(() -> reader.search(query, 5, 16)).get().ordinals().length);
      executor
          .submit(
              () -> {
                reader.close();
                return null;
              })
          .get();
    }
    reader.close();
    assertThrows(IllegalStateException.class, () -> reader.search(query, 5, 16));
  }

  private static final int COUNT = 256, DIMENSION = 2;

  /** Writes a 2-D fixture, a ring graph (or self-loops only) over it, and prepares an index. */
  private Path prepareRing(String name, boolean connected) throws IOException {
    Path data = directory.resolve(name + "-vectors.bin");
    ByteBuffer vectors =
        ByteBuffer.allocate(8 + COUNT * DIMENSION * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    vectors.putInt(COUNT).putInt(DIMENSION);
    for (int i = 0; i < COUNT; i++) vectors.putFloat(i).putFloat(i % 17);
    Files.write(data, vectors.array());

    Path graph = directory.resolve(name + "-graph.bin");
    ByteBuffer edges =
        ByteBuffer.allocate(24 + COUNT * 2 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    edges.putLong(edges.capacity()).putInt(1).putInt(0).putLong(0);
    for (int i = 0; i < COUNT; i++) edges.putInt(1).putInt(connected ? (i + 1) % COUNT : i);
    Files.write(graph, edges.array());

    Path prefix = directory.resolve(name);
    DiskAnnIndex.prepare(data, graph, prefix, 1);
    return prefix;
  }

  private static void overwriteInt(Path path, long offset, int value) throws IOException {
    try (FileChannel file = FileChannel.open(path, StandardOpenOption.WRITE)) {
      file.write(
          ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(value).flip(),
          offset);
    }
  }

  private static void overwriteLong(Path path, long offset, long value) throws IOException {
    try (FileChannel file = FileChannel.open(path, StandardOpenOption.WRITE)) {
      file.write(
          ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(value).flip(),
          offset);
    }
  }
}
