/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.DiskAnnIndex;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Objects;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;

/** Experimental flat flushes and controlled, single-graph merges. */
public class DiskAnnVectorsFormat extends KnnVectorsFormat {
  private final Lucene99FlatVectorsFormat flat =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);
  private final Options options;

  /** SPI reader with conservative defaults. Graph construction must be enabled explicitly. */
  public DiskAnnVectorsFormat() {
    this("CuVSDiskAnn", Options.defaults());
  }

  public DiskAnnVectorsFormat(Options options) {
    this("CuVSDiskAnn", options);
  }

  protected DiskAnnVectorsFormat(String name, Options options) {
    super(name);
    this.options = Objects.requireNonNull(options);
  }

  public record Options(
      Path scratch,
      DiskAnnGraphBuilder.Limits limits,
      int pqChunks,
      long readerCacheBytes,
      int searchList,
      boolean buildOnForceMerge) {
    public Options {
      Objects.requireNonNull(scratch);
      Objects.requireNonNull(limits);
      if (pqChunks < 0 || pqChunks > 128 || readerCacheBytes < 1 || searchList < 1)
        throw new IllegalArgumentException("Invalid DiskANN format options");
    }

    public static Options defaults() {
      return new Options(
          Path.of(System.getProperty("java.io.tmpdir")),
          new DiskAnnGraphBuilder.Limits(256, 4096, 256L << 20),
          0,
          2L << 30,
          128,
          false);
    }
  }

  @Override
  public int getMaxDimensions(String field) {
    return 128;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Writer(state, flat.fieldsWriter(state));
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Reader(state, flat.fieldsReader(state));
  }

  private static IllegalArgumentException unsupported(String feature) {
    return new IllegalArgumentException("PrototypeDiskAnn does not support " + feature);
  }

  private static String prefix(String segment, String suffix, int field) {
    return segment + "_" + suffix + "_" + field;
  }

  private static final String ARTIFACT_CODEC = "PrototypeDiskAnnArtifact";

  public static void extract(
      Directory source, String name, Directory destination, byte[] segmentId, String suffix)
      throws IOException {
    try (var input = source.openChecksumInput(name);
        var output = destination.createOutput(name, IOContext.DEFAULT)) {
      CodecUtil.checkIndexHeader(input, ARTIFACT_CODEC, 0, 0, segmentId, suffix);
      long payloadLength = input.length() - input.getFilePointer() - CodecUtil.footerLength();
      if (payloadLength < 0)
        throw new CorruptIndexException("Invalid graph artifact length", input);
      output.copyBytes(input, payloadLength);
      CodecUtil.checkFooter(input);
    }
  }

  private static void validate(FieldInfo field) {
    if (field.getVectorEncoding() != VectorEncoding.FLOAT32
        || field.getVectorSimilarityFunction() != VectorSimilarityFunction.EUCLIDEAN
        || field.getVectorDimension() > 128)
      throw unsupported("non-float32/L2 fields or dimensions above 128");
  }

  private final class Writer extends KnnVectorsWriter {
    private final SegmentWriteState state;
    private final FlatVectorsWriter delegate;
    private FieldInfo field;
    private int count;

    Writer(SegmentWriteState state, FlatVectorsWriter delegate) {
      this.state = state;
      this.delegate = delegate;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo info) throws IOException {
      select(info);
      @SuppressWarnings("unchecked")
      var target = (KnnFieldVectorsWriter<float[]>) delegate.addField(info);
      return new KnnFieldVectorsWriter<float[]>() {
        @Override
        public void addValue(int docId, float[] value) throws IOException {
          if (value.length != info.getVectorDimension()) throw unsupported("dimension mismatch");
          for (float v : value) if (!Float.isFinite(v)) throw unsupported("non-finite vectors");
          target.addValue(docId, value);
          count++;
        }

        @Override
        public float[] copyValue(float[] value) {
          return value.clone();
        }

        @Override
        public long ramBytesUsed() {
          return target.ramBytesUsed();
        }
      };
    }

    private void select(FieldInfo info) {
      validate(info);
      if (field != null && field != info)
        throw unsupported("multiple fields in one format instance");
      field = info;
    }

    @Override
    public long ramBytesUsed() {
      return delegate.ramBytesUsed();
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
      if (sortMap != null) throw unsupported("index sorting");
      delegate.flush(maxDoc, null);
      if (field != null) metadata(null);
    }

    @Override
    public void mergeOneField(FieldInfo info, MergeState merge) throws IOException {
      select(info);
      if (merge.needsIndexSort) throw unsupported("index sorting");
      delegate.mergeOneField(info, merge);
      count = 0;
      var iterator = MergedVectorValues.mergeFloatVectorValues(info, merge).iterator();
      while (iterator.nextDoc() != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS) count++;
      // Lucene supplies this public context for forced merges. Routine merges
      // must not start GPU construction or inherit its experimental size limits.
      if (!options.buildOnForceMerge()
          || state.context.mergeInfo() == null
          || state.context.mergeInfo().mergeMaxNumSegments() != 1
          || count < 256
          || count > options.limits().maxGraphRows()) {
        metadata(null);
        return;
      }
      Path root = options.scratch();
      Files.createDirectories(root);
      Path scratch = Files.createTempDirectory(root, "lucene-vamana-");
      Throwable primary = null;
      try {
        var limits = options.limits();
        var spool =
            DiskAnnGraphBuilder.spool(
                MergedVectorValues.mergeFloatVectorValues(info, merge), scratch, limits);
        count = spool.count();
        var built = DiskAnnGraphBuilder.build(spool, scratch, limits, options.pqChunks());
        metadata(built);
        if (state.infoStream.isEnabled("CuVSDiskAnn"))
          state.infoStream.message(
              "CuVSDiskAnn",
              "rows="
                  + built.count()
                  + " subgraphs="
                  + built.subgraphs()
                  + " largest="
                  + built.largestSubgraph()
                  + " matrixBytes="
                  + built.matrixBytes()
                  + " scratchBytes="
                  + built.scratchBytes());
      } catch (IOException | RuntimeException | Error e) {
        primary = e;
        throw e;
      } finally {
        try {
          IOUtils.rm(scratch);
        } catch (IOException e) {
          if (primary == null) throw e;
          primary.addSuppressed(e);
        }
      }
    }

    private void metadata(DiskAnnGraphBuilder.Built built) throws IOException {
      String base = prefix(state.segmentInfo.name, state.segmentSuffix, field.number);
      String[] suffixes = built == null ? new String[0] : built.suffixes();
      long[] lengths = new long[suffixes.length], checksums = new long[suffixes.length];
      for (int i = 0; i < suffixes.length; i++) {
        try (var input = Files.newInputStream(Path.of(built.prefix() + suffixes[i]));
            var output = state.directory.createOutput(base + suffixes[i], state.context)) {
          CodecUtil.writeIndexHeader(
              output, ARTIFACT_CODEC, 0, state.segmentInfo.getId(), state.segmentSuffix);
          output.copyBytes(
              new InputStreamDataInput(input), Files.size(Path.of(built.prefix() + suffixes[i])));
          // Compound files require a Lucene header as well as a footer.
          CodecUtil.writeFooter(output);
          lengths[i] = output.getFilePointer();
        }
        try (var input = state.directory.openInput(base + suffixes[i], IOContext.READONCE)) {
          checksums[i] = CodecUtil.retrieveChecksum(input);
        }
      }
      try (var meta = state.directory.createOutput(base + ".dkm", state.context)) {
        CodecUtil.writeIndexHeader(
            meta, "PrototypeDiskAnn", 3, state.segmentInfo.getId(), state.segmentSuffix);
        meta.writeInt(field.number);
        meta.writeInt(field.getVectorDimension());
        meta.writeInt(count);
        meta.writeInt(suffixes.length);
        for (int i = 0; i < suffixes.length; i++) {
          meta.writeString(suffixes[i]);
          meta.writeLong(lengths[i]);
          meta.writeLong(checksums[i]);
        }
        CodecUtil.writeFooter(meta);
      }
    }

    @Override
    public void finish() throws IOException {
      delegate.finish();
    }

    @Override
    public void close() throws IOException {
      delegate.close();
    }
  }

  private final class Reader extends KnnVectorsReader {
    private final SegmentReadState state;
    private final FlatVectorsReader delegate;
    private DiskAnnIndex index;
    private Path extracted;
    private FieldInfo field;
    private String base;
    private int count;
    private String[] suffixes = new String[0];
    private long[] lengths = new long[0], checksums = new long[0];

    Reader(SegmentReadState state, FlatVectorsReader delegate) throws IOException {
      this.state = state;
      this.delegate = delegate;
      boolean success = false;
      try {
        for (FieldInfo candidate : state.fieldInfos)
          if (candidate.hasVectorValues()
              && getName()
                  .equals(candidate.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY))
              && state.segmentSuffix.endsWith(
                  getName()
                      + "_"
                      + candidate.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_SUFFIX_KEY))) {
            if (field != null) throw unsupported("multiple vector fields");
            validate(candidate);
            field = candidate;
          }
        if (field == null) throw new IOException("Missing vector field");
        base = prefix(state.segmentInfo.name, state.segmentSuffix, field.number);
        try (var meta = state.directory.openChecksumInput(base + ".dkm")) {
          CodecUtil.checkIndexHeader(
              meta, "PrototypeDiskAnn", 3, 3, state.segmentInfo.getId(), state.segmentSuffix);
          if (meta.readInt() != field.number || meta.readInt() != field.getVectorDimension())
            throw new CorruptIndexException("Field metadata mismatch", meta);
          count = meta.readInt();
          if (count < 0 || delegate.getFloatVectorValues(field.name).size() != count)
            throw new CorruptIndexException("Vector count mismatch", meta);
          int n = meta.readInt();
          if (n != 0 && n != 5) throw new CorruptIndexException("Invalid artifact count", meta);
          suffixes = new String[n];
          lengths = new long[n];
          checksums = new long[n];
          var allowed =
              new HashSet<>(
                  java.util.List.of(
                      "_disk.index",
                      "_disk.index_medoids.bin",
                      "_pq_pivots.bin",
                      "_pq_compressed.bin",
                      ".ordmap"));
          for (int i = 0; i < n; i++) {
            suffixes[i] = meta.readString();
            if (!allowed.remove(suffixes[i]))
              throw new CorruptIndexException("Invalid artifact name", meta);
            lengths[i] = meta.readLong();
            checksums[i] = meta.readLong();
          }
          CodecUtil.checkFooter(meta);
        }
        // Validate framing first. Native cache extraction below checks payloads
        // while copying, avoiding a separate full-file validation pass.
        for (int i = 0; i < suffixes.length; i++) {
          try (var input = state.directory.openInput(base + suffixes[i], IOContext.READONCE)) {
            if (input.length() != lengths[i] || CodecUtil.retrieveChecksum(input) != checksums[i])
              throw new CorruptIndexException("DiskANN artifact length/footer mismatch", input);
            input.seek(0);
            CodecUtil.checkIndexHeader(
                input, ARTIFACT_CODEC, 0, 0, state.segmentInfo.getId(), state.segmentSuffix);
          }
        }
        if (suffixes.length > 0) {
          try (var map = state.directory.openChecksumInput(base + ".ordmap")) {
            var values = delegate.getFloatVectorValues(field.name);
            CodecUtil.checkIndexHeader(
                map, ARTIFACT_CODEC, 0, 0, state.segmentInfo.getId(), state.segmentSuffix);
            long headerLength = map.getFilePointer();
            if (map.readInt() != count
                || map.readInt() != 1
                || map.length() != headerLength + 8L + 4L * count + CodecUtil.footerLength())
              throw new CorruptIndexException("Ordinal mapping header mismatch", map);
            for (int ord = 0; ord < count; ord++)
              if (map.readInt() != values.ordToDoc(ord))
                throw new CorruptIndexException("Ordinal mapping mismatch", map);
            CodecUtil.checkFooter(map);
          }
          // DiskANN requires raw paths, while Lucene owns header/footer-wrapped
          // files (possibly inside a compound file). Keep the raw cache private.
          long bytes = 0;
          for (long length : lengths) bytes = Math.addExact(bytes, length);
          long budget = options.readerCacheBytes();
          if (budget < 1 || bytes > budget)
            throw new IOException("Graph reader cache exceeds budget");
          Path root = options.scratch();
          Files.createDirectories(root);
          extracted = Files.createTempDirectory(root, "graph-reader-");
          try (var destination = FSDirectory.open(extracted)) {
            for (String suffix : suffixes)
              if (!suffix.equals(".ordmap"))
                extract(
                    state.directory,
                    base + suffix,
                    destination,
                    state.segmentInfo.getId(),
                    state.segmentSuffix);
          }
          index = DiskAnnIndex.open(extracted.resolve(base), field.getVectorDimension(), count);
        }
        success = true;
      } finally {
        if (!success) IOUtils.closeWhileHandlingException(this);
      }
    }

    @Override
    public void checkIntegrity() throws IOException {
      delegate.checkIntegrity();
      for (int i = 0; i < suffixes.length; i++) {
        try (var input = state.directory.openChecksumInput(base + suffixes[i])) {
          if (input.length() != lengths[i] || input.length() < CodecUtil.footerLength())
            throw new CorruptIndexException("DiskANN artifact length mismatch", input);
          input.skipBytes(input.length() - CodecUtil.footerLength());
          if (CodecUtil.checkFooter(input) != checksums[i])
            throw new CorruptIndexException("DiskANN artifact checksum mismatch", input);
        }
      }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String name) throws IOException {
      return delegate.getFloatVectorValues(name);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String name) {
      throw unsupported("byte vectors");
    }

    @Override
    public void search(String name, byte[] query, KnnCollector collector, Bits acceptDocs) {
      throw unsupported("byte vectors");
    }

    @Override
    public void search(String name, float[] query, KnnCollector collector, Bits acceptDocs)
        throws IOException {
      if (!field.name.equals(name))
        throw new IllegalArgumentException("Unknown vector field " + name);
      var values = delegate.getFloatVectorValues(name);
      // Native filtering/visit accounting is not implemented. Use Lucene's
      // existing exact scorer for these cases, including normal live-doc masks.
      if (index == null || acceptDocs != null || collector.visitLimit() < Integer.MAX_VALUE) {
        var scorer = delegate.getRandomVectorScorer(name, query);
        for (int ord = 0; ord < count && !collector.earlyTerminated(); ord++) {
          int doc = values.ordToDoc(ord);
          if (acceptDocs == null || acceptDocs.get(doc)) {
            collector.collect(doc, scorer.score(ord));
            collector.incVisitedCount(1);
          }
        }
        return;
      }
      int k = Math.min(collector.k(), count);
      if (k == 0) return;
      int searchList = options.searchList();
      if (searchList < 1) throw new IllegalArgumentException("Search list must be positive");
      var result = index.search(query, k, Math.min(count, Math.max(k, searchList)));
      var seen = new HashSet<Long>();
      for (int i = 0; i < k; i++) {
        long ord = result.ordinals()[i];
        float distance = result.distances()[i];
        if (ord < 0 || ord >= count || !seen.add(ord) || !Float.isFinite(distance) || distance < 0)
          throw new IOException("Invalid DiskANN search result");
        collector.collect(values.ordToDoc((int) ord), 1.0f / (1.0f + distance));
        collector.incVisitedCount(1);
      }
    }

    @Override
    public void close() throws IOException {
      IOUtils.close(
          index,
          delegate,
          () -> {
            if (extracted != null) IOUtils.rm(extracted);
          });
    }
  }
}
