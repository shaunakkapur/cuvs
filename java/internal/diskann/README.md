# Experimental DiskANN reader

This cuVS fork owns the native reader and single-graph PQ/layout conversion
bridge. Java callers use `com.nvidia.cuvs.DiskAnnIndex`. Microsoft DiskANN is a
provisional dependency, not an upstream cuVS adoption decision.

The optional target uses Microsoft DiskANN at
`78256bbab4685e1774e78d331e081a153be26823` with the three patches in `patches/`.
They bound PQ encoding buffers, replace the known Linux AIO process-exit path, and
reject short search results before copying them. The synchronous I/O patch
changes performance, so these runs are compatibility evidence, not stock DiskANN
latency results. The API accepts trusted generated float32/L2 files only.

Prepare a PIC DiskANN archive with those patches and a dependency environment
containing Boost, libaio, MKL and tcmalloc. No sudo or cuVS rebuild is needed for
this CPU-only target. With JDK 22 in `JAVA_HOME`, configure and build it using
the paths to that environment and archive.

```sh
cmake -S java/internal/diskann -B java/internal/diskann/build \
  -DDISKANN_SOURCE="$DISKANN_SOURCE" -DDISKANN_ARCHIVE="$DISKANN_ARCHIVE" \
  -DDISKANN_DEPS="$DISKANN_DEPS" -DCMAKE_BUILD_TYPE=Release
cmake --build java/internal/diskann/build --parallel 2
```

It is also available as `BUILD_DISKANN_JAVA=ON` in `java/internal`. The default
build remains unchanged. Load `libcuvs_diskann.so` explicitly with
`DiskAnnIndex.loadLibrary(path)` or through the JVM library path. Library loading
and native errors become `IOException`; JVM fatal errors are not suppressed.

This bridge remains JNI to preserve the already tested boundary while moving it
into the library. It is not an all-Java DiskANN implementation or a new standalone
utility in the prototype. Converting it to cuVS's generated C/FFM binding path is
separate API work, not required to validate the repository split.

Run the optional CPU-only reader tests with `mvn -f java/cuvs-java/pom.xml
-Dit.test=DiskAnnIndexIT -Ddiskann.library=/absolute/path/libcuvs_diskann.so verify`.
Without the library property these optional tests are skipped. No GPU is used.
The local artifact is `cuvs-java:26.12.0-diskann-reader-SNAPSHOT`, not a Maven
Central release. Stitching and Lucene integration are separate changes.
