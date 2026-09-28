# DiskANN vectors format

`DiskAnnVectorsFormat` stores float32 EUCLIDEAN vectors in Lucene's flat format and
can add a DiskANN graph per field. cuVS builds the graph on the GPU; queries search
it on the CPU through `com.nvidia.cuvs.DiskAnnIndex` (see `c/diskann/README.md`).

Flushes and ordinary merges write flat vectors. A graph is built during a forced
merge to one segment when `Options.buildOnForceMerge()` is set and the field has
256 to `Options.maxGraphRows()` vectors (default 4,096). Otherwise, or if the build
fails, the field stays flat; failures are logged to the `CuVSDiskAnn` info stream.
The no-argument format used to open segments by name never builds graphs.

Unfiltered queries on fields with a graph use it. Filtered queries, segments with
deletions, visit limits and flat fields are scored exactly. Graph files are normal
segment files with codec headers and footers. Because the native reader needs plain
files, opening a segment copies them to a temporary directory under
`Options.scratch()`, removed on close.

Limits: at most 128 dimensions, one graph per field per segment, and searches on one
segment's graph are serialized.

`TestDiskAnnFormatDefaults` and `TestDiskAnnFlatMerges` need no GPU. Run all tests,
including the GPU tests in `DiskAnnCodecIT`, with
`DISKANN_TEST_GPU=ON bash ci/test_diskann.sh`.
