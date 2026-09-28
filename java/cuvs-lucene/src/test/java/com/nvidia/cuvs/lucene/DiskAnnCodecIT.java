/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.*;

import com.nvidia.cuvs.DiskAnnIndex;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Bits;
import org.junit.BeforeClass;
import org.junit.Test;

public class DiskAnnCodecIT extends LuceneTestCase {
  static Path temporary;
  static Path index;
  static final float[][] VECTORS = new float[1024][16];

  @BeforeClass
  public static void buildThroughLucene() throws Exception {
    temporary = createTempDir();
    GraphTestFormat.scratch = createTempDir();
    String library = System.getProperty("diskann.library");
    if (library == null)
      throw new IllegalStateException("Set diskann.library for the diskann test profile");
    DiskAnnIndex.loadLibrary(Path.of(library));
    index = temporary.resolve("index");
    Random random = new Random(731);
    var config =
        new IndexWriterConfig()
            .setUseCompoundFile(false)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMergeScheduler(new SerialMergeScheduler())
            .setCodec(
                new Lucene101Codec() {
                  @Override
                  public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new GraphTestFormat();
                  }
                });
    try (var directory = FSDirectory.open(index);
        var writer = new IndexWriter(directory, config)) {
      for (int i = 0; i < VECTORS.length; i++) {
        // Gaps force graph ordinals to differ from Lucene document IDs.
        if (i % 7 == 0) {
          Document gap = new Document();
          gap.add(new StoredField("source_id", "gap-" + i));
          writer.addDocument(gap);
        }
        for (int d = 0; d < 16; d++) VECTORS[i][d] = random.nextFloat() * 5 - 2.3f;
        Document doc = new Document();
        doc.add(new StoredField("source_id", "source-" + (10000 - i * 3)));
        doc.add(new StringField("key", "row-" + i, Field.Store.NO));
        doc.add(new KnnFloatVectorField("vector", VECTORS[i], VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc);
        if (i % 300 == 299) writer.commit();
      }
      writer.commit();
      try (var old = DirectoryReader.open(writer)) {
        assertTrue(old.leaves().size() > 1);

        assertEquals(
            10,
            new IndexSearcher(old)
                .search(new KnnFloatVectorQuery("vector", VECTORS[0], 10), 10)
                .scoreDocs
                .length);

        var mergePolicy = new TieredMergePolicy();
        mergePolicy.setNoCFSRatio(0);
        writer.getConfig().setMergePolicy(mergePolicy);
        writer.forceMerge(1);
        writer.commit();
        // A retained source reader remains searchable after replacement publication.
        assertEquals(
            10,
            new IndexSearcher(old)
                .search(new KnnFloatVectorQuery("vector", VECTORS[0], 10), 10)
                .scoreDocs
                .length);
      }
    }
  }

  @Test
  public void reopenedReaderUsesDiskAnnAndPreservesDocumentIdentity() throws Exception {

    int recalled = 0;
    Random random = new Random(619);
    // Two independent opens, after the writer, cuVS resources and scratch files are closed.
    for (int reopen = 0; reopen < 2; reopen++) {
      try (var directory = FSDirectory.open(index);
          var reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        var leaf = reader.leaves().getFirst().reader();
        var values = leaf.getFloatVectorValues("vector");
        var probe = leaf.searchNearestVectors("vector", VECTORS[0], 10, null, Integer.MAX_VALUE);
        assertEquals(
            "Unfiltered search must use the graph, not a full scan", 10, probe.totalHits.value());
        int[] documentToOrdinal = new int[leaf.maxDoc()];
        Arrays.fill(documentToOrdinal, -1);
        boolean hasGap = false;
        for (int ord = 0; ord < values.size(); ord++) {
          int doc = values.ordToDoc(ord);
          documentToOrdinal[doc] = ord;
          hasGap |= doc != ord;
        }
        assertTrue("Fixture must exercise sparse document IDs", hasGap);
        for (int q = 0; q < 50; q++) {
          float[] query = new float[16];
          for (int d = 0; d < query.length; d++) query[d] = random.nextFloat() * 5 - 2.3f;
          double[] truth = new double[VECTORS.length];
          for (int i = 0; i < truth.length; i++) truth[i] = distance(query, VECTORS[i]);
          double[] sorted = truth.clone();
          Arrays.sort(sorted);
          var result =
              new IndexSearcher(reader).search(new KnnFloatVectorQuery("vector", query, 10), 10);
          var hits = result.scoreDocs;
          assertTrue(result.totalHits.value() >= hits.length);
          assertEquals(10, hits.length);
          Set<Integer> seen = new HashSet<>();
          float previous = Float.POSITIVE_INFINITY;
          for (var hit : hits) {
            assertTrue(seen.add(hit.doc));
            String id = leaf.storedFields().document(hit.doc).get("source_id");
            assertTrue(id.startsWith("source-"));
            int ordinal = (10000 - Integer.parseInt(id.substring(7))) / 3;
            // Tiered merging may reorder source segments. Source identity is
            // independent of the final graph ordinal and merged document ID.
            int graphOrdinal = documentToOrdinal[hit.doc];
            assertTrue(graphOrdinal >= 0);
            assertEquals(hit.doc, values.ordToDoc(graphOrdinal));
            assertArrayEquals(VECTORS[ordinal], values.vectorValue(graphOrdinal), 0f);
            assertEquals(1 / (1 + truth[ordinal]), hit.score, 1e-5);
            assertTrue(hit.score <= previous);
            previous = hit.score;
            if (truth[ordinal] <= sorted[9]) recalled++;
          }
        }
      }
    }

    double recall = recalled / 1000.0;
    assertTrue("Recall@10 = " + recall, recall >= 0.95);
  }

  @Test
  public void segmentOwnsArtifactsAndScratchIsGone() throws Exception {
    try (var directory = FSDirectory.open(index)) {
      var segments = SegmentInfos.readLatestCommit(directory);
      var files = segments.info(0).info.files();
      for (String extension : GRAPH_EXTENSIONS)
        assertEquals(1, files.stream().filter(name -> hasExtension(name, extension)).count());
      for (String file : files) {
        assertTrue(Files.isRegularFile(index.resolve(file)));
        try (var input = directory.openInput(file, IOContext.READONCE)) {
          CodecUtil.checksumEntireFile(input);
        }
      }
    }
    try (var files = Files.list(GraphTestFormat.scratch)) {
      assertEquals(
          0, files.filter(p -> p.getFileName().toString().startsWith("cuvs-diskann-")).count());
    }
  }

  @Test
  public void closedIndexCanMoveWithoutItsOriginalBuildPath() throws Exception {
    Path destination = Files.createDirectory(temporary.resolve("relocated"));
    copyIndex(destination);

    try (var directory = FSDirectory.open(destination);
        var reader = DirectoryReader.open(directory)) {
      assertEquals(
          10,
          new IndexSearcher(reader)
              .search(new KnnFloatVectorQuery("vector", VECTORS[0], 10), 10)
              .scoreDocs
              .length);
    }
  }

  @Test
  public void missingAndCorruptArtifactsFailBeforeSearch() throws Exception {
    for (String extension : GRAPH_EXTENSIONS) {
      for (String fault : new String[] {"missing", "payload", "footer"}) {
        Path copy = Files.createTempDirectory(temporary, fault);
        copyIndex(copy);
        Path target;
        try (var files = Files.list(copy)) {
          target =
              files
                  .filter(p -> hasExtension(p.getFileName().toString(), extension))
                  .findFirst()
                  .orElseThrow();
        }
        if (fault.equals("missing")) Files.delete(target);
        else {
          byte[] data = Files.readAllBytes(target);
          data[fault.equals("payload") ? 0 : data.length - 1] ^= 1;
          Files.write(target, data);
        }
        try (var directory = FSDirectory.open(copy)) {
          assertThrows(
              IOException.class,
              () -> {
                try (var reader = DirectoryReader.open(directory)) {
                  for (var leaf : reader.leaves()) leaf.reader().checkIntegrity();
                }
              });
        }
      }
    }
  }

  @Test
  public void filtersAndVisitLimitsUseTheExactFallback() throws Exception {
    try (var directory = FSDirectory.open(index);
        var reader = DirectoryReader.open(directory)) {
      var leaf = reader.leaves().getFirst().reader();

      assertEquals(
          0,
          leaf.searchNearestVectors(
                  "vector", VECTORS[0], 10, new Bits.MatchNoBits(leaf.maxDoc()), Integer.MAX_VALUE)
              .scoreDocs
              .length);
      assertTrue(
          leaf.searchNearestVectors("vector", VECTORS[0], 10, null, 1).scoreDocs.length <= 1);
    }
  }

  @Test
  public void deletedAndUpdatedDocumentsStaySearchableWithoutRebuildingTheGraph() throws Exception {
    Path copy = Files.createTempDirectory(temporary, "mutations");
    copyIndex(copy);
    var config = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE);
    try (var directory = FSDirectory.open(copy);
        var writer = new IndexWriter(directory, config)) {
      writer.deleteDocuments(new Term("key", "row-0"));
      Document replacement = new Document();
      replacement.add(new StringField("key", "row-1", Field.Store.YES));
      replacement.add(new StoredField("source_id", "replacement"));
      replacement.add(
          new KnnFloatVectorField("vector", VECTORS[0], VectorSimilarityFunction.EUCLIDEAN));
      writer.updateDocument(new Term("key", "row-1"), replacement);
      writer.commit();
    }
    try (var directory = FSDirectory.open(copy);
        var reader = DirectoryReader.open(directory)) {
      assertTrue("Exercise a graph segment with live-doc exclusions", reader.hasDeletions());
      var searcher = new IndexSearcher(reader);
      var hits = searcher.search(new KnnFloatVectorQuery("vector", VECTORS[0], 10), 10).scoreDocs;
      assertEquals(10, hits.length);
      assertEquals("replacement", searcher.storedFields().document(hits[0].doc).get("source_id"));
      for (var hit : hits) {
        String id = searcher.storedFields().document(hit.doc).get("source_id");
        assertNotEquals("source-10000", id);
        assertNotEquals("source-9997", id);
      }
    }
  }

  @Test
  public void forcedGraphMergeIntoCompoundFilesReopensAndCleansItsReaderCache() throws Exception {
    Path copy = Files.createTempDirectory(temporary, "compound");
    var policy = new TieredMergePolicy();
    policy.setNoCFSRatio(1.0);
    var config =
        new IndexWriterConfig()
            .setUseCompoundFile(true)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMergeScheduler(new SerialMergeScheduler())
            .setCodec(
                new Lucene101Codec() {
                  @Override
                  public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new GraphTestFormat();
                  }
                });
    try (var directory = FSDirectory.open(copy);
        var writer = new IndexWriter(directory, config)) {
      for (int i = 0; i < 512; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vector", VECTORS[i], VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc);
        if (i == 255) writer.commit();
      }
      writer.commit();
      writer.getConfig().setMergePolicy(policy);
      writer.forceMerge(1);
      writer.commit();
      assertTrue(SegmentInfos.readLatestCommit(directory).info(0).info.getUseCompoundFile());
    }

    for (int attempt = 0; attempt < 2; attempt++) {
      try (var directory = FSDirectory.open(copy);
          var reader = DirectoryReader.open(directory)) {
        reader.leaves().getFirst().reader().checkIntegrity();
        assertEquals(
            10,
            new IndexSearcher(reader)
                .search(new KnnFloatVectorQuery("vector", VECTORS[0], 10), 10)
                .scoreDocs
                .length);
      }
    }

    try (var files = Files.list(GraphTestFormat.scratch)) {
      assertEquals(
          0, files.filter(p -> p.getFileName().toString().startsWith("cuvs-diskann-")).count());
    }
  }

  /** DiskANN files plus the metadata file, by extension prefix (field numbers follow). */
  static final String[] GRAPH_EXTENSIONS = {"dki", "dkd", "dkp", "dkc", "dkm"};

  static boolean hasExtension(String file, String prefix) {
    String extension = IndexFileNames.getExtension(file);
    return extension != null && extension.startsWith(prefix);
  }

  private static void copyIndex(Path destination) throws Exception {
    try (var source = FSDirectory.open(index)) {
      for (String file : SegmentInfos.readLatestCommit(source).files(true))
        Files.copy(index.resolve(file), destination.resolve(file));
    }
  }

  static double distance(float[] a, float[] b) {
    double sum = 0;
    for (int i = 0; i < a.length; i++) {
      double diff = (double) a[i] - b[i];
      sum += diff * diff;
    }
    return sum;
  }
}
