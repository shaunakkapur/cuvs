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
    org.junit.Assume.assumeNotNull(System.getProperty("diskann.library"));
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
    assertThrows(IOException.class, () -> new DiskAnnIndex(prefix, 16, 256));
    assertThrows(IllegalArgumentException.class, () -> new DiskAnnIndex(prefix, 0, 256));
    assertThrows(IllegalArgumentException.class, () -> new DiskAnnIndex(prefix, 16, 1_000_001));
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
    int count = 256, dimension = 2;
    Path data = directory.resolve("vectors.bin");
    ByteBuffer vectors =
        ByteBuffer.allocate(8 + count * dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    vectors.putInt(count).putInt(dimension);
    for (int i = 0; i < count; i++) vectors.putFloat(i).putFloat(i % 17);
    Files.write(data, vectors.array());

    Path graph = directory.resolve("graph.bin");
    ByteBuffer edges =
        ByteBuffer.allocate(24 + count * 2 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    edges.putLong(edges.capacity()).putInt(1).putInt(0).putLong(0);
    for (int i = 0; i < count; i++) edges.putInt(1).putInt((i + 1) % count);
    Files.write(graph, edges.array());

    Path prefix = directory.resolve("read-failure");
    DiskAnnIndex.prepare(data, graph, prefix, 1);
    try (DiskAnnIndex reader = new DiskAnnIndex(prefix, dimension, count)) {
      DiskAnnIndex.Results results = reader.search(new float[] {42, 8}, 5, count);
      assertEquals(5, results.ordinals().length);
      assertEquals(5, results.distances().length);
    }
    Path diskIndex = directory.resolve("read-failure_disk.index");
    overwriteLong(diskIndex, 8, count + 1L);
    IOException invalidDisk =
        assertThrows(IOException.class, () -> new DiskAnnIndex(prefix, dimension, count));
    assertTrue(invalidDisk.getMessage().contains("Invalid disk index metadata"));
    overwriteLong(diskIndex, 8, count);

    Path compressed = directory.resolve("read-failure_pq_compressed.bin");
    overwriteInt(compressed, 0, 1_000_000);
    IOException invalidCompressed =
        assertThrows(IOException.class, () -> new DiskAnnIndex(prefix, dimension, count));
    assertTrue(invalidCompressed.getMessage().contains("Invalid matrix file"));
    overwriteInt(compressed, 0, count);

    Path pivots = directory.resolve("read-failure_pq_pivots.bin");
    overwriteInt(pivots, 4096, 255);
    IOException invalidPivots =
        assertThrows(IOException.class, () -> new DiskAnnIndex(prefix, dimension, count));
    assertTrue(invalidPivots.getMessage().contains("Invalid PQ pivot metadata"));
    overwriteInt(pivots, 4096, 256);

    try (DiskAnnIndex reader = new DiskAnnIndex(prefix, dimension, count);
        FileChannel disk =
            FileChannel.open(
                directory.resolve("read-failure_disk.index"), StandardOpenOption.WRITE)) {
      disk.truncate(0);
      assertThrows(IOException.class, () -> reader.search(new float[] {42, 8}, 5, count));
    }
  }

  @Test
  public void reportsWhenTheGraphCannotProduceEnoughResults() throws Exception {
    int count = 256, dimension = 2;
    Path data = directory.resolve("disconnected-vectors.bin");
    ByteBuffer vectors =
        ByteBuffer.allocate(8 + count * dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    vectors.putInt(count).putInt(dimension);
    for (int i = 0; i < count; i++) vectors.putFloat(i).putFloat(i % 17);
    Files.write(data, vectors.array());

    Path graph = directory.resolve("disconnected-graph.bin");
    ByteBuffer edges =
        ByteBuffer.allocate(24 + count * 2 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    edges.putLong(edges.capacity()).putInt(1).putInt(0).putLong(0);
    for (int i = 0; i < count; i++) edges.putInt(1).putInt(i);
    Files.write(graph, edges.array());

    Path prefix = directory.resolve("disconnected-index");
    DiskAnnIndex.prepare(data, graph, prefix, 1);
    try (DiskAnnIndex reader = new DiskAnnIndex(prefix, dimension, count)) {
      IOException failure =
          assertThrows(IOException.class, () -> reader.search(new float[] {42, 8}, 5, count));
      assertTrue(failure.getMessage().contains("fewer results than requested"));
    }
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
