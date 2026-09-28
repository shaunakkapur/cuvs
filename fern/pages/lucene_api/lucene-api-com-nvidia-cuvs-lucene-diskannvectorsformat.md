---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-diskannvectorsformat
---

# DiskAnnVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class DiskAnnVectorsFormat extends KnnVectorsFormat
```

Flat float32 EUCLIDEAN vectors plus an optional DiskANN graph per field, built by cuVS during a
forced merge to one segment and searched on the CPU through `DiskAnnIndex`. See
DISKANN.md in this module for when graphs are built and how queries use them.

## Public Members

### defaults

```java
public static Options defaults()
```

Reads existing graphs but never builds new ones. Graphs of up to 4096 vectors, 8 PQ chunks
(or fewer for small dimensions), a search list of 128, and scratch under \{@code
java.io.tmpdir\}.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/DiskAnnVectorsFormat.java:88`_

### DiskAnnVectorsFormat

```java
public DiskAnnVectorsFormat()
```

Reader-only format with `Options#defaults()`, used when Lucene loads it by name.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/DiskAnnVectorsFormat.java:94`_

### extract

```java
static void extract( Directory source, String name, Path destination, byte[] segmentId, String suffix) throws IOException
```

Copies a graph file's payload to `destination`, verifying its header and checksum.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/DiskAnnVectorsFormat.java:134`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/DiskAnnVectorsFormat.java:52`_
