/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.nio.file.Path;

/**
 * Test format that builds graphs on forced merges. Registered by name so readers opened by the
 * tests use the same scratch directory; tests set {@link #scratch} before indexing.
 */
public final class GraphTestFormat extends DiskAnnVectorsFormat {
  static final String NAME = "CuVSDiskAnnTest";
  static Path scratch = Path.of(System.getProperty("java.io.tmpdir"));

  public GraphTestFormat() {
    super(NAME, new Options(scratch, 4096, 0, 128, true));
  }
}
