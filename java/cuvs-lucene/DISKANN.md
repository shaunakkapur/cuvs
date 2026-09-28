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

This slice builds one graph for 256..4,096 vectors. Smaller or larger controlled
merges stay flat. Bounded clustered stitching is a separate dependent change;
this is not a large-partition construction result.

Reader options are constructor-local. Lucene's named-format SPI uses the no-arg
reader defaults unless the host registers its own format subclass with the same
configuration, as the prototype fixture does. No machine path is persisted.
The stock defaults are a 2 GiB per-reader raw cache under `java.io.tmpdir` and a
search list of 128. Raw copies are required by the native path-based reader and
are removed on normal close or failed open; process crashes can leave copies.
There is no shared-cache quota or crash scavenger yet.

Deletes and filters remain correct through exact fallback, not native filtered
ANN. Those queries scan the segment and are not a 5B-scale performance result.
Float32, squared L2, up to 128 dimensions and 4,096 vectors per graph are
the current single-graph limits. The Microsoft DiskANN C API and generated Java
bindings live in this fork. They are not published upstream dependencies.

CPU-only library regression tests use the existing Maven/Lucene framework:

```sh
mvn -f java/cuvs-lucene/pom.xml -Dtest=TestDiskAnnLibrary test
```

The opt-in `DiskAnnCodecIT` tests live in this module and cover GPU construction,
document identity, compound files, filters, deletes, corruption and reopening.
Run `DISKANN_TEST_GPU=ON bash ci/test_diskann.sh` from the repository root, after
installing the reader dependency. See `c/diskann/README.md` for prerequisites.
The explicit `diskann` Maven profile fails if its native library is unavailable.
The OpenSearch harness is a separate, test-only consumer of this codec.
