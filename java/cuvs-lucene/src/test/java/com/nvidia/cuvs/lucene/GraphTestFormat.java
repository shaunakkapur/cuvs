/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.nio.file.Path;

/** Test SPI reader; all graph construction is explicitly enabled by the writer. */
public final class GraphTestFormat extends DiskAnnVectorsFormat {
  static Path scratch;

  public GraphTestFormat() {
    super(
        "PrototypeDiskAnn",
        new Options(
            scratch == null ? Path.of(System.getProperty("java.io.tmpdir")) : scratch,
            new DiskAnnGraphBuilder.Limits(256, 4096, 256L << 20),
            0,
            2L << 30,
            128,
            true));
  }
}
