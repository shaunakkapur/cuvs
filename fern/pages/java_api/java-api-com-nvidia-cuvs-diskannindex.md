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

### loadLibrary

```java
static void loadLibrary(Path library) throws IOException
```

Load the optional C library from an explicit path before opening readers.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:25`_

### prepare

```java
static String[] prepare( Path data, Path graph, Path prefix, int pqChunks, int sampleRows, int blockRows) throws IOException
```

Bounded plain-PQ conversion; returns the files produced relative to prefix.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:34`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/DiskAnnIndex.java:16`_
