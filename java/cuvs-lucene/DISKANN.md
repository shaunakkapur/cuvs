# Experimental partition graphs

`DiskAnnVectorsFormat` owns graph artifacts through Lucene `Directory`, including
standard headers/footers, compound files and merge-abort cleanup. The single-graph
builder uses cuvs-java for GPU Vamana construction and the fork's `DiskAnnIndex`
for CPU PQ/layout conversion and disk search.

Fresh segments and ordinary merges stay flat. The default format does not build
graphs, even during `forceMerge(1)`. A host must opt in with `Options` before a
controlled force merge after bulk ingestion. This needs multiple source segments;
a force merge of one already-clean segment may do no work. A later ordinary merge
can produce a flat segment again. This is not a continuous-ingestion lifecycle.

This slice builds one graph for at most 4,096 vectors. Larger input is rejected
before GPU construction. Bounded clustered stitching is a separate dependent
change; this is not a large-partition construction result.

Reader options are constructor-local. Lucene's named-format SPI uses the no-arg
reader defaults unless the host registers its own format subclass with the same
configuration, as the prototype fixture does. No machine path is persisted.
The stock defaults are a 2 GiB per-reader raw cache under `java.io.tmpdir` and a
search list of 128. Raw copies are required by the native path-based reader and
are removed on normal close or failed open; process crashes can leave copies.
There is no shared-cache quota or crash scavenger yet.

Deletes and filters remain correct through exact fallback, not native filtered
ANN. Those queries scan the segment and are not a 5B-scale performance result.
Float32, squared L2, up to 128 dimensions and 1M vectors per built partition are
the current tested feasibility envelope. Microsoft DiskANN and the JNI bridge
are provisional fork dependencies, not upstream integration commitments.

CPU-only library regression tests use the existing Maven/Lucene framework:

```sh
mvn -f java/cuvs-lucene/pom.xml -Dtest=TestDiskAnnLibrary test
```

The prototype keeps the opt-in GPU, document-identity, compound-file, reopened
reader and embedded OpenSearch installation tests. Its plugin consumes this JAR;
it does not implement the codec or native reader.

Build with `java/internal/diskann/setup.sh`. This produces
`cuvs-java:26.12.0-diskann-reader-SNAPSHOT` and
`cuvs-lucene:26.12.0-diskann-codec-SNAPSHOT`. The prototype compatibility harness
consumes these two drafts. The CPU-only `TestDiskAnnLibrary` tests check safe
defaults and rejection before GPU work.
