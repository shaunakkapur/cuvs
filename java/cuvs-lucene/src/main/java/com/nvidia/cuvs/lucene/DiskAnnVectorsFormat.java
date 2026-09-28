/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.DiskAnnIndex;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.InputStreamDataInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;

/**
 * Flat float32 EUCLIDEAN vectors plus an optional DiskANN graph per field, built by cuVS during a
 * forced merge to one segment and searched on the CPU through {@link DiskAnnIndex}. See
 * DISKANN.md in this module for when graphs are built and how queries use them.
 */
public class DiskAnnVectorsFormat extends KnnVectorsFormat {
  static final String NAME = "CuVSDiskAnn";
  static final String META_CODEC = "CuVSDiskAnnMeta";
  static final String DATA_CODEC = "CuVSDiskAnnData";
  static final String META_EXTENSION = "dkm";
  static final int VERSION = 0;
  static final int MAX_DIMENSIONS = 128;

  /** Lucene file extension prefix for each file written by {@link DiskAnnIndex#prepare}. */
  private static final Map<String, String> EXTENSIONS =
      Map.of(
          "_disk.index", "dki",
          "_disk.index_medoids.bin", "dkd",
          "_pq_pivots.bin", "dkp",
          "_pq_compressed.bin", "dkc");

  private final Lucene99FlatVectorsFormat flat =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);
  private final Options options;

  /** Settings for graph construction and search. */
  public record Options(
      Path scratch, int maxGraphRows, int pqChunks, int searchList, boolean buildOnForceMerge) {
    public Options {
      Objects.requireNonNull(scratch);
      if (maxGraphRows < DiskAnnGraphBuilder.MIN_ROWS
          || maxGraphRows > 1_000_000
          || pqChunks < 0
          || pqChunks > MAX_DIMENSIONS
          || searchList < 1) throw new IllegalArgumentException("Invalid DiskANN format options");
    }

    /**
     * Reads existing graphs but never builds new ones. Graphs of up to 4096 vectors, 8 PQ chunks
     * (or fewer for small dimensions), a search list of 128, and scratch under {@code
     * java.io.tmpdir}.
     */
    public static Options defaults() {
      return new Options(Path.of(System.getProperty("java.io.tmpdir")), 4096, 0, 128, false);
    }
  }

  /** Reader-only format with {@link Options#defaults()}, used when Lucene loads it by name. */
  public DiskAnnVectorsFormat() {
    this(NAME, Options.defaults());
  }

  public DiskAnnVectorsFormat(Options options) {
    this(NAME, options);
  }

  protected DiskAnnVectorsFormat(String name, Options options) {
    super(name);
    this.options = Objects.requireNonNull(options);
  }

  @Override
  public int getMaxDimensions(String field) {
    return MAX_DIMENSIONS;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Writer(state, flat.fieldsWriter(state));
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Reader(state, flat.fieldsReader(state));
  }

  private static String fileName(String segment, String suffix, String nativeFile, int field) {
    return IndexFileNames.segmentFileName(segment, suffix, EXTENSIONS.get(nativeFile) + field);
  }

  private static void checkSupported(FieldInfo field) {
    if (field.getVectorEncoding() != VectorEncoding.FLOAT32
        || field.getVectorSimilarityFunction() != VectorSimilarityFunction.EUCLIDEAN)
      throw new IllegalArgumentException(
          NAME + " supports only float32 vectors with EUCLIDEAN similarity: " + field.name);
  }

  /** Copies a graph file's payload to {@code destination}, verifying its header and checksum. */
  static void extract(
      Directory source, String name, Path destination, byte[] segmentId, String suffix)
      throws IOException {
    try (var input = source.openChecksumInput(name);
        var output = Files.newOutputStream(destination)) {
      CodecUtil.checkIndexHeader(input, DATA_CODEC, VERSION, VERSION, segmentId, suffix);
      long remaining = input.length() - input.getFilePointer() - CodecUtil.footerLength();
      if (remaining < 0) throw new CorruptIndexException("Truncated graph file", input);
      byte[] buffer = new byte[1 << 16];
      while (remaining > 0) {
        int n = (int) Math.min(buffer.length, remaining);
        input.readBytes(buffer, 0, n);
        output.write(buffer, 0, n);
        remaining -= n;
      }
      CodecUtil.checkFooter(input);
    }
  }

  private final class Writer extends KnnVectorsWriter {
    private final SegmentWriteState state;
    private final FlatVectorsWriter delegate;

    /** Field number to whether the field has a graph in this segment. */
    private final Map<Integer, Boolean> fields = new LinkedHashMap<>();

    Writer(SegmentWriteState state, FlatVectorsWriter delegate) {
      this.state = state;
      this.delegate = delegate;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo field) throws IOException {
      checkSupported(field);
      fields.put(field.number, false);
      return delegate.addField(field);
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
      delegate.flush(maxDoc, sortMap);
    }

    @Override
    public void mergeOneField(FieldInfo field, MergeState mergeState) throws IOException {
      checkSupported(field);
      delegate.mergeOneField(field, mergeState);
      var merge = state.context.mergeInfo();
      boolean forcedToOne = merge != null && merge.mergeMaxNumSegments() == 1;
      fields.put(
          field.number,
          options.buildOnForceMerge()
              && forcedToOne
              && !mergeState.needsIndexSort
              && writeGraph(field, mergeState));
    }

    /** Builds and stores a graph for the field. Returns false if the field should stay flat. */
    private boolean writeGraph(FieldInfo field, MergeState mergeState) throws IOException {
      Files.createDirectories(options.scratch());
      Path scratch = Files.createTempDirectory(options.scratch(), "cuvs-diskann-build-");
      try {
        Path prefix = scratch.resolve("graph");
        try {
          var values = MergedVectorValues.mergeFloatVectorValues(field, mergeState);
          Path data = scratch.resolve("vectors");
          int count = DiskAnnGraphBuilder.spool(values, data, options.maxGraphRows());
          if (count < DiskAnnGraphBuilder.MIN_ROWS) {
            log(
                "field "
                    + field.name
                    + " stays flat: "
                    + (count < 0 ? "too many" : count)
                    + " vectors");
            return false;
          }
          int chunks =
              options.pqChunks() == 0 ? Math.min(8, values.dimension()) : options.pqChunks();
          DiskAnnGraphBuilder.build(
              data, count, values.dimension(), prefix, Math.min(chunks, values.dimension()));
        } catch (IOException | RuntimeException | LinkageError e) {
          // A failed build must not fail the merge; the field stays searchable, exactly.
          log("field " + field.name + " stays flat: graph build failed: " + e);
          return false;
        }
        for (String nativeFile : DiskAnnIndex.PREPARED_FILES) {
          String name =
              fileName(state.segmentInfo.name, state.segmentSuffix, nativeFile, field.number);
          try (var input = Files.newInputStream(Path.of(prefix + nativeFile));
              var output = state.directory.createOutput(name, state.context)) {
            CodecUtil.writeIndexHeader(
                output, DATA_CODEC, VERSION, state.segmentInfo.getId(), state.segmentSuffix);
            output.copyBytes(
                new InputStreamDataInput(input), Files.size(Path.of(prefix + nativeFile)));
            CodecUtil.writeFooter(output);
          }
        }
        return true;
      } finally {
        IOUtils.rm(scratch);
      }
    }

    private void log(String message) {
      if (state.infoStream.isEnabled(NAME)) state.infoStream.message(NAME, message);
    }

    @Override
    public void finish() throws IOException {
      delegate.finish();
      String name =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
      try (IndexOutput meta = state.directory.createOutput(name, state.context)) {
        CodecUtil.writeIndexHeader(
            meta, META_CODEC, VERSION, state.segmentInfo.getId(), state.segmentSuffix);
        for (var entry : fields.entrySet()) {
          meta.writeInt(entry.getKey());
          meta.writeByte((byte) (entry.getValue() ? 1 : 0));
        }
        meta.writeInt(-1);
        CodecUtil.writeFooter(meta);
      }
    }

    @Override
    public long ramBytesUsed() {
      return delegate.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
      delegate.close();
    }
  }

  private final class Reader extends KnnVectorsReader {
    private final SegmentReadState state;
    private final FlatVectorsReader delegate;
    private final Map<String, DiskAnnIndex> graphs = new HashMap<>();
    private final List<String> graphFiles = new ArrayList<>();
    private Path extracted;

    Reader(SegmentReadState state, FlatVectorsReader delegate) throws IOException {
      this.state = state;
      this.delegate = delegate;
      boolean success = false;
      try {
        for (FieldInfo field : readGraphFields()) {
          if (extracted == null) {
            Files.createDirectories(options.scratch());
            extracted = Files.createTempDirectory(options.scratch(), "cuvs-diskann-reader-");
          }
          Path prefix = extracted.resolve("f" + field.number);
          for (String nativeFile : DiskAnnIndex.PREPARED_FILES) {
            String name =
                fileName(state.segmentInfo.name, state.segmentSuffix, nativeFile, field.number);
            graphFiles.add(name);
            extract(
                state.directory,
                name,
                Path.of(prefix + nativeFile),
                state.segmentInfo.getId(),
                state.segmentSuffix);
          }
          int count = delegate.getFloatVectorValues(field.name).size();
          graphs.put(field.name, DiskAnnIndex.open(prefix, field.getVectorDimension(), count));
        }
        success = true;
      } finally {
        if (!success) IOUtils.closeWhileHandlingException(this);
      }
    }

    private List<FieldInfo> readGraphFields() throws IOException {
      String name =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
      var result = new ArrayList<FieldInfo>();
      try (var meta = state.directory.openChecksumInput(name)) {
        Throwable failure = null;
        try {
          CodecUtil.checkIndexHeader(
              meta, META_CODEC, VERSION, VERSION, state.segmentInfo.getId(), state.segmentSuffix);
          for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
            FieldInfo field = state.fieldInfos.fieldInfo(number);
            if (field == null) throw new CorruptIndexException("Invalid field " + number, meta);
            if (meta.readByte() != 0) result.add(field);
          }
        } catch (Throwable t) {
          failure = t;
        } finally {
          CodecUtil.checkFooter(meta, failure);
        }
      }
      return result;
    }

    @Override
    public void checkIntegrity() throws IOException {
      delegate.checkIntegrity();
      for (String name : graphFiles) {
        try (IndexInput input = state.directory.openInput(name, IOContext.READONCE)) {
          CodecUtil.checksumEntireFile(input);
        }
      }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
      return delegate.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) {
      throw new UnsupportedOperationException(NAME + " does not support byte vectors");
    }

    @Override
    public void search(String field, byte[] target, KnnCollector collector, Bits acceptDocs) {
      throw new UnsupportedOperationException(NAME + " does not support byte vectors");
    }

    @Override
    public void search(String field, float[] target, KnnCollector collector, Bits acceptDocs)
        throws IOException {
      var values = delegate.getFloatVectorValues(field);
      DiskAnnIndex graph = graphs.get(field);
      // The native reader does not filter or count visits, so use the exact scorer for those.
      if (graph == null || acceptDocs != null || collector.visitLimit() < Integer.MAX_VALUE) {
        var scorer = delegate.getRandomVectorScorer(field, target);
        for (int ord = 0; ord < values.size() && !collector.earlyTerminated(); ord++) {
          int doc = values.ordToDoc(ord);
          if (acceptDocs == null || acceptDocs.get(doc)) {
            collector.collect(doc, scorer.score(ord));
            collector.incVisitedCount(1);
          }
        }
        return;
      }
      int k = Math.min(collector.k(), values.size());
      if (k == 0) return;
      var result =
          graph.search(target, k, Math.min(values.size(), Math.max(k, options.searchList())));
      for (int i = 0; i < k; i++) {
        // DiskANN returns squared L2 distances; this matches Lucene's EUCLIDEAN score.
        collector.collect(
            values.ordToDoc((int) result.ordinals()[i]), 1.0f / (1.0f + result.distances()[i]));
        collector.incVisitedCount(1);
      }
    }

    @Override
    public void close() throws IOException {
      var closeables = new ArrayList<Closeable>(graphs.values());
      closeables.add(delegate);
      closeables.add(
          () -> {
            if (extracted != null) IOUtils.rm(extracted);
          });
      IOUtils.close(closeables);
    }
  }
}
