/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/** Experimental bridge for trusted generated files; the codec validates persisted artifacts. */
public final class DiskAnnIndex implements Closeable {
  private static boolean libraryLoaded;

  private final int dimension;
  private final int count;
  private long handle;

  public record Results(long[] ordinals, float[] distances) {}

  public static void prepare(Path data, Path graph, Path prefix, int pqChunks) throws IOException {
    ensureLibraryLoaded();
    nativeCall(
        () -> {
          prepare0(
              data.toAbsolutePath().toString(),
              graph.toAbsolutePath().toString(),
              prefix.toAbsolutePath().toString(),
              pqChunks);
          return null;
        });
  }

  public static String[] prepare(
      Path data, Path graph, Path prefix, int pqChunks, int sampleLimit, int blockRows)
      throws IOException {
    ensureLibraryLoaded();
    Objects.requireNonNull(data, "data");
    Objects.requireNonNull(graph, "graph");
    Objects.requireNonNull(prefix, "prefix");
    return nativeCall(
        () ->
            prepareBounded0(
                data.toAbsolutePath().toString(),
                graph.toAbsolutePath().toString(),
                prefix.toAbsolutePath().toString(),
                pqChunks,
                sampleLimit,
                blockRows));
  }

  public DiskAnnIndex(Path prefix, int dimension, int expectedCount) throws IOException {
    ensureLibraryLoaded();
    if (dimension < 1 || dimension > 128 || expectedCount < 256 || expectedCount > 1_000_000)
      throw new IllegalArgumentException("Index exceeds reader limits");
    this.dimension = dimension;
    this.count = expectedCount;
    handle = nativeCall(() -> open0(prefix.toAbsolutePath().toString(), dimension, expectedCount));
  }

  public synchronized Results search(float[] query, int k, int searchList) throws IOException {
    if (handle == 0) throw new IllegalStateException("Reader is closed");
    if (query.length != dimension || k < 1 || k > count || searchList < k || searchList > count)
      throw new IllegalArgumentException("Invalid query dimensions or search budget");
    for (float value : query)
      if (!Float.isFinite(value)) throw new IllegalArgumentException("Non-finite query");
    long[] ordinals = new long[k];
    float[] distances = new float[k];
    nativeCall(
        () -> {
          search0(handle, query, searchList, ordinals, distances);
          return null;
        });
    return new Results(ordinals, distances);
  }

  @Override
  public synchronized void close() throws IOException {
    if (handle != 0) {
      long previous = handle;
      handle = 0;
      nativeCall(
          () -> {
            close0(previous);
            return null;
          });
    }
  }

  private static synchronized void ensureLibraryLoaded() throws IOException {
    if (libraryLoaded) return;
    try {
      System.loadLibrary("cuvs_diskann");
    } catch (UnsatisfiedLinkError error) {
      throw new IOException(
          "Cannot load the optional cuVS DiskANN library; build java/internal/diskann", error);
    }
    libraryLoaded = true;
  }

  public static synchronized void loadLibrary(Path library) throws IOException {
    try {
      System.load(library.toAbsolutePath().toString());
      libraryLoaded = true;
    } catch (UnsatisfiedLinkError error) {
      throw new IOException("Cannot load the configured DiskANN JNI library", error);
    }
  }

  static <T> T nativeCall(IOSupplier<T> operation) throws IOException {
    try {
      return operation.get();
    } catch (UnsatisfiedLinkError error) {
      throw new IOException("DiskANN JNI API mismatch; rebuild java/internal/diskann", error);
    }
  }

  @FunctionalInterface
  private interface IOSupplier<T> {
    T get() throws IOException;
  }

  private static native void prepare0(String data, String graph, String prefix, int chunks)
      throws IOException;

  private static native String[] prepareBounded0(
      String data, String graph, String prefix, int chunks, int sampleLimit, int blockRows)
      throws IOException;

  private static native long open0(String prefix, int dimension, int count) throws IOException;

  private static native void search0(
      long handle, float[] query, int list, long[] ids, float[] distances) throws IOException;

  private static native void close0(long handle) throws IOException;
}
