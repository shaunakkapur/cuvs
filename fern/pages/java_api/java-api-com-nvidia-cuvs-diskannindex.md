---
slug: api-reference/java-api-com-nvidia-cuvs-diskannindex
---

# DiskAnnIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface DiskAnnIndex extends Closeable
```

Optional CPU DiskANN reader. Searches on one reader are serialized; distinct readers are
independent.

## Public Members

### of

```java
List<String> PREPARED_FILES = List.of("_disk.index", "_disk.index_medoids.bin", "_pq_pivots.bin", "_pq_compressed.bin")
```

Files written by `#prepare`, as suffixes of its output prefix.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:18`_

### loadLibrary

```java
static void loadLibrary(Path library) throws IOException
```

Load the optional C library from an explicit path before opening readers.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:30`_

### prepare

```java
static void prepare( Path data, Path graph, Path prefix, int pqChunks, int sampleRows, int blockRows) throws IOException
```

Converts a serialized cuVS Vamana graph and its vectors into a disk index with plain PQ,
writing `#PREPARED_FILES` under `prefix`. PQ training uses at most \{@code
sampleRows\} vectors and encoding reads `blockRows` vectors at a time.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:43`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:17`_
