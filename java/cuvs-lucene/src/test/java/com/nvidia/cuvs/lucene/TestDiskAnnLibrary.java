/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;

/** CPU-only defaults and policy validation. No native library is required. */
public class TestDiskAnnLibrary extends LuceneTestCase {
  public void testForcedMergeDoesNotImplicitlyEnableGraphConstruction() throws Exception {
    var config =
        new IndexWriterConfig()
            .setUseCompoundFile(true)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMergeScheduler(new SerialMergeScheduler())
            .setCodec(
                new Lucene101Codec() {
                  @Override
                  public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new DiskAnnVectorsFormat();
                  }
                });
    try (var directory = FSDirectory.open(createTempDir());
        var writer = new IndexWriter(directory, config)) {
      for (int i = 0; i < 1700; i++) {
        var doc = new Document();
        doc.add(
            new KnnFloatVectorField(
                "vector", new float[] {i, i * .1f}, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc);
        if (i % 400 == 399) writer.commit();
      }
      writer.commit();
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
      writer.commit();
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        assertEquals(1700, reader.numDocs());
        reader.leaves().getFirst().reader().checkIntegrity();
        var hits =
            new IndexSearcher(reader)
                .search(new KnnFloatVectorQuery("vector", new float[] {0, 0}, 5), 5);
        assertEquals(5, hits.scoreDocs.length);
        assertEquals(1f, hits.scoreDocs[0].score, 0f);
      }
      for (String file : directory.listAll()) assertFalse(file.endsWith("_disk.index"));
    }
  }

  public void testOversizedInputIsRejectedBeforeGpuConstruction() throws Exception {
    var rows = new ArrayList<float[]>();
    for (int i = 0; i < 300; i++) rows.add(new float[] {i, .1f * i});
    var limits = new DiskAnnGraphBuilder.Limits(256, 256, 128L << 20);
    var scratch = createTempDir();
    expectThrows(
        IOException.class,
        () -> DiskAnnGraphBuilder.spool(FloatVectorValues.fromFloats(rows, 2), scratch, limits));
    assertFalse(Files.exists(scratch.resolve("vectors.bin")));
  }

  public void testSingleGraphConfiguration() {
    var limits = new DiskAnnGraphBuilder.Limits(256, 400, 128L << 20);
    assertEquals(400, DiskAnnGraphBuilder.pqBlockRows(limits));
    assertFalse(DiskAnnVectorsFormat.Options.defaults().buildOnForceMerge());
    expectThrows(
        IllegalArgumentException.class,
        () -> new DiskAnnGraphBuilder.Limits(256, 4097, 128L << 20));
    expectThrows(
        IllegalArgumentException.class,
        () -> new DiskAnnVectorsFormat.Options(createTempDir(), limits, 1, 0, 128, true));
  }
}
