# Experimental DiskANN C API

Linux x86_64 CPU reader for float32 squared-L2 indexes. This optional fork component
uses cuVS's existing DiskANN CPM dependency, pinned in `cpp/cmake/patches/diskann_override.json`.
The C API lives in `c/include/cuvs/neighbors/diskann.h`. Java uses generated Panama
bindings through the standard provider, not JNI or a prototype-owned native build.

Activate the cuVS environment and supply MKL, Boost, libaio and gperftools development
packages. Run `bash c/diskann/build.sh` (CMake 4+, Ninja). An existing dependency prefix
may be supplied with `DISKANN_DEPS`. No environment is created automatically.
This builds the CPU dependency and reader, not libcuvs. Reuse `CPM_SOURCE_CACHE` if set.
`bash ci/test_diskann.sh` builds the reader, stages headers, generates bindings and
runs its CPU tests. Maven honors `MAVEN_ARGS` for a repository path or mirror.
This is also the optional fork-CI entrypoint; upstream's existing Java job does not
install the additional CPU-reader dependencies or run this profile automatically.

Run CPU reader checks with `mvn -f java/cuvs-java/pom.xml -Pdiskann
-Dit.test=DiskAnnIndexIT -Ddiskann.library=/absolute/path/libcuvs_diskann.so verify`.
The explicit profile fails if its library is missing; normal GPU-only builds do not
load or test this optional backend. Fork CI needs a runner with these dependencies
and the same profile, not an assumption that existing upstream CI covers it.

Searches on one handle are serialized, and the native reader currently uses synchronous
pread to turn I/O failures into errors rather than exiting the process. This is a
correctness/compatibility backend, not a DiskANN throughput benchmark. PQ codes stay
in memory. Artifacts must be trusted, with the codec validating persisted checksums.
The recorded patches bound PQ buffers and reject short results safely. They are fork
dependency changes, not claims of upstream DiskANN fixes.
