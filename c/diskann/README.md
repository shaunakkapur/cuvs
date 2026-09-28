# DiskANN C API

Optional Linux x86_64 CPU reader for float32, squared-L2 DiskANN disk indexes, and
conversion of a serialized cuVS Vamana graph into that disk layout with plain PQ.
The API is declared in `c/include/cuvs/neighbors/diskann.h` and exposed to Java
through generated Panama bindings (`com.nvidia.cuvs.DiskAnnIndex`).

The library is built separately from libcuvs because it needs CPU dependencies the
GPU build does not: MKL, Boost, libaio and gperftools. It reuses cuVS's DiskANN
dependency (`cpp/cmake/thirdparty/get_diskann.cmake`) at a pinned commit, with three
patches listed there:

- bounded PQ encoding blocks, so preparation memory does not grow with the index;
- an error instead of an out-of-bounds read when a search finds fewer than k results;
- synchronous reads that throw on I/O errors instead of calling `exit()`.

## Build and test

With a cuVS development environment active and the dependencies above installed
(`DISKANN_DEPS` may point to an extra prefix that contains them):

```sh
bash c/diskann/build.sh
bash ci/test_diskann.sh
```

`ci/test_diskann.sh` builds the library, generates the bindings and runs the CPU
reader tests (`DiskAnnIndexIT`) with the `diskann` Maven profile. It installs into
`MAVEN_LOCAL_REPO` (default `c/diskann/build/m2`) so it does not replace a
`cuvs-java` artifact in `~/.m2`. The default Maven build skips these tests.

## Limits

- 256 to 1,000,000 vectors and at most 128 dimensions.
- Searches on one index handle are serialized, and reads are synchronous. Use one
  handle per concurrent reader; this is not a throughput-optimized path.
- PQ codes stay in memory. Index files are trusted input.
