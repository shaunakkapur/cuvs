/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.*;

import java.nio.file.Path;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** No native library or GPU required for ordinary Lucene operations. */
public class TestDiskAnnFlatMerges extends LuceneTestCase {
  Path temporary;

  @org.junit.Before
  public void setUpPaths() {
    temporary = createTempDir();
    GraphTestFormat.scratch = createTempDir();
  }

  @Test
  public void ordinaryMergesWithDefaultCompoundFilesAndTwoFieldsDoNotBuildGraphs()
      throws Exception {
    var policy = new LogDocMergePolicy();
    policy.setMergeFactor(2);
    policy.setMinMergeDocs(32);
    policy.setNoCFSRatio(1.0);

    try (var directory = FSDirectory.open(temporary);
        var writer = new IndexWriter(directory, config(policy))) {
      for (int i = 0; i < 3000; i++) {
        writer.addDocument(document(i));
        if (i % 100 == 99) writer.commit();
      }
      writer.maybeMerge();
      writer.commit();
      try (var reader = DirectoryReader.open(writer)) {
        assertEquals(3000, reader.numDocs());
        assertTrue("Routine merges must actually run", reader.leaves().size() < 30);
        assertTrue(reader.leaves().stream().anyMatch(leaf -> leaf.reader().maxDoc() > 1350));
        checkFields(reader);
      }
      assertTrue(
          SegmentInfos.readLatestCommit(directory).asList().stream()
              .anyMatch(info -> info.info.getUseCompoundFile()));
    }
    try (var directory = FSDirectory.open(temporary);
        var reader = DirectoryReader.open(directory)) {
      checkFields(reader);
    }
  }

  @Test
  public void smallForcedMergeStaysFlatBelowPqTrainingMinimum() throws Exception {

    try (var directory = FSDirectory.open(temporary);
        var writer = new IndexWriter(directory, config(NoMergePolicy.INSTANCE))) {
      for (int i = 0; i < 20; i++) {
        writer.addDocument(document(i));
        if (i == 9) writer.commit();
      }
      writer.commit();
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
      writer.commit();
      try (var reader = DirectoryReader.open(writer)) {
        assertEquals(1, reader.leaves().size());
        checkFields(reader);
      }
    }
  }

  @Test
  public void oversizedForcedMergeStaysFlatAndSearchable() throws Exception {
    try (var directory = FSDirectory.open(temporary);
        var writer = new IndexWriter(directory, config(NoMergePolicy.INSTANCE))) {
      for (int i = 0; i < 4200; i++) {
        writer.addDocument(document(i));
        if (i % 1000 == 999) writer.commit();
      }
      writer.commit();
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
      writer.commit();
      try (var reader = DirectoryReader.open(writer)) {
        assertEquals(1, reader.leaves().size());
        checkFields(reader);
        assertEquals(4200, reader.numDocs());
      }
      for (String file : directory.listAll()) assertFalse(file.contains(".dki"));
    }
  }

  private static IndexWriterConfig config(MergePolicy policy) {
    // One format instance serves both fields, as a codec normally returns.
    var format = new GraphTestFormat();
    return new IndexWriterConfig()
        .setUseCompoundFile(true)
        .setMergePolicy(policy)
        .setMergeScheduler(new SerialMergeScheduler())
        .setCodec(
            new Lucene101Codec() {
              @Override
              public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return format;
              }
            });
  }

  private static Document document(int i) {
    var doc = new Document();
    doc.add(
        new KnnFloatVectorField(
            "first", new float[] {i * .1f, .3f}, VectorSimilarityFunction.EUCLIDEAN));
    doc.add(
        new KnnFloatVectorField(
            "second", new float[] {.2f, i * .1f}, VectorSimilarityFunction.EUCLIDEAN));
    return doc;
  }

  private static void checkFields(DirectoryReader reader) throws Exception {
    var searcher = new IndexSearcher(reader);
    for (String field : new String[] {"first", "second"}) {
      assertEquals(
          10,
          searcher.search(new KnnFloatVectorQuery(field, new float[] {.2f, .3f}, 10), 10)
              .scoreDocs
              .length);
    }
    for (var leaf : reader.leaves()) leaf.reader().checkIntegrity();
  }
}
