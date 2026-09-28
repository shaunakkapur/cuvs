/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.panama.diskann.diskann_h.*;
import static java.lang.foreign.ValueLayout.*;

import com.nvidia.cuvs.DiskAnnIndex;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Objects;

/** Generated C-API bindings, isolated from the optional library until first use. */
public final class DiskAnnIndexImpl implements DiskAnnIndex {
  private final int dimension;
  private final int count;
  private long handle;
  private static boolean libraryLoaded;

  public DiskAnnIndexImpl(Path prefix, int dimension, int count) throws IOException {
    Objects.requireNonNull(prefix);
    if (dimension < 1 || dimension > 128 || count < 256 || count > 1_000_000)
      throw new IllegalArgumentException("Index exceeds reader limits");
    ensureLibraryLoaded();
    this.dimension = dimension;
    this.count = count;
    try (var arena = Arena.ofConfined()) {
      var out = arena.allocate(JAVA_LONG);
      check(
          cuvsDiskAnnOpen(
              arena.allocateFrom(prefix.toAbsolutePath().toString()), dimension, count, out));
      handle = out.get(JAVA_LONG, 0);
    }
  }

  public static synchronized void loadLibrary(Path library) throws IOException {
    try {
      System.load(Objects.requireNonNull(library).toAbsolutePath().toString());
      validateBindings();
      libraryLoaded = true;
    } catch (LinkageError error) {
      throw new IOException("Cannot load the optional cuVS DiskANN C library", error);
    }
  }

  private static synchronized void ensureLibraryLoaded() throws IOException {
    if (libraryLoaded) return;
    try {
      System.loadLibrary("cuvs_diskann");
      validateBindings();
      libraryLoaded = true;
    } catch (LinkageError error) {
      throw new IOException("Build the optional C library with c/diskann/build.sh", error);
    }
  }

  private static void validateBindings() {
    // Resolve symbols before entering a Lucene merge, reporting setup failures as IO errors.
    cuvsDiskAnnOpen$address();
    cuvsDiskAnnSearch$address();
    cuvsDiskAnnClose$address();
    cuvsDiskAnnPrepare$address();
  }

  public static void prepare(Path data, Path graph, Path prefix, int chunks, int sample, int block)
      throws IOException {
    ensureLibraryLoaded();
    try (var arena = Arena.ofConfined()) {
      check(
          cuvsDiskAnnPrepare(
              arena.allocateFrom(data.toAbsolutePath().toString()),
              arena.allocateFrom(graph.toAbsolutePath().toString()),
              arena.allocateFrom(prefix.toAbsolutePath().toString()),
              chunks,
              sample,
              block));
    }
  }

  @Override
  public synchronized Results search(float[] query, int k, int searchList) throws IOException {
    if (handle == 0) throw new IllegalStateException("Reader is closed");
    if (query.length != dimension || k < 1 || k > count || searchList < k || searchList > count)
      throw new IllegalArgumentException("Invalid query dimensions or search budget");
    for (float value : query)
      if (!Float.isFinite(value)) throw new IllegalArgumentException("Non-finite query");
    try (var arena = Arena.ofConfined()) {
      var ids = arena.allocate(JAVA_LONG, k);
      var distances = arena.allocate(JAVA_FLOAT, k);
      check(
          cuvsDiskAnnSearch(
              handle,
              arena.allocateFrom(JAVA_FLOAT, query),
              dimension,
              k,
              searchList,
              ids,
              distances));
      return new Results(ids.toArray(JAVA_LONG), distances.toArray(JAVA_FLOAT));
    }
  }

  @Override
  public synchronized void close() throws IOException {
    if (handle == 0) return;
    long previous = handle;
    handle = 0;
    check(cuvsDiskAnnClose(previous));
  }

  static void check(int status) throws IOException {
    if (status != CUVS_SUCCESS()) {
      var error = cuvsGetLastErrorText.makeInvoker().apply();
      throw new IOException(
          error.address() == 0
              ? "DiskANN C API failed"
              : error.reinterpret(Long.MAX_VALUE).getString(0));
    }
  }
}
