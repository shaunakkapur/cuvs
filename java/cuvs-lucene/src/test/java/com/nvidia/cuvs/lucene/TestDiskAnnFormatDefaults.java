/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

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

/** Defaults and option validation. No native library or GPU is required. */
public class TestDiskAnnFormatDefaults extends LuceneTestCase {
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
      for (String file : directory.listAll()) assertFalse(file.contains(".dki"));
    }
  }

  public void testSpoolStopsAfterTheRowLimit() throws Exception {
    var rows = new ArrayList<float[]>();
    for (int i = 0; i < 300; i++) rows.add(new float[] {i, .1f * i});
    var values = FloatVectorValues.fromFloats(rows, 2);
    assertEquals(-1, DiskAnnGraphBuilder.spool(values, createTempDir().resolve("over"), 299));
    assertEquals(300, DiskAnnGraphBuilder.spool(values, createTempDir().resolve("ok"), 300));
  }

  public void testOptionValidation() {
    assertFalse(DiskAnnVectorsFormat.Options.defaults().buildOnForceMerge());
    var scratch = createTempDir();
    new DiskAnnVectorsFormat.Options(scratch, 1_000_000, 0, 128, true);
    expectThrows(
        IllegalArgumentException.class,
        () -> new DiskAnnVectorsFormat.Options(scratch, 255, 0, 128, true));
    expectThrows(
        IllegalArgumentException.class,
        () -> new DiskAnnVectorsFormat.Options(scratch, 4096, 129, 128, true));
    expectThrows(
        IllegalArgumentException.class,
        () -> new DiskAnnVectorsFormat.Options(scratch, 4096, 0, 0, true));
  }
}
