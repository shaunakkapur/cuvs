---
slug: api-reference/c-api-neighbors-diskann
---

# Diskann

_Source header: `cuvs/neighbors/diskann.h`_

## Experimental DiskANN CPU reader

<a id="cuvsdiskannindex-t"></a>
### cuvsDiskAnnIndex_t

Handle for trusted float32, squared-L2 DiskANN artifacts.

```c
typedef uintptr_t cuvsDiskAnnIndex_t;
```

<a id="cuvsdiskannopen"></a>
### cuvsDiskAnnOpen

Open an index with 256..1M rows and 1..128 dimensions.

```c
cuvsError_t cuvsDiskAnnOpen(const char* prefix,
uint32_t dimension,
uint32_t count,
cuvsDiskAnnIndex_t* handle);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `prefix` |  | `const char*` | Path prefix shared by disk-layout and PQ files. |
| `dimension` |  | `uint32_t` | Expected vector dimension. |
| `count` |  | `uint32_t` | Expected number of vectors. |
| `handle` |  | [`cuvsDiskAnnIndex_t*`](/api-reference/c-api-neighbors-diskann#cuvsdiskannindex-t) | Receives the owned handle; close it with cuvsDiskAnnClose. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsdiskannsearch"></a>
### cuvsDiskAnnSearch

Search a loaded index using squared L2 distance.

```c
cuvsError_t cuvsDiskAnnSearch(cuvsDiskAnnIndex_t handle,
const float* query,
uint32_t dimension,
uint32_t k,
uint32_t search_list,
uint64_t* ids,
float* distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | [`cuvsDiskAnnIndex_t`](/api-reference/c-api-neighbors-diskann#cuvsdiskannindex-t) | An open reader handle. |
| `query` |  | `const float*` | Finite float32 query with dimension elements. |
| `dimension` |  | `uint32_t` | Dimension used when opening the reader. |
| `k` |  | `uint32_t` | Number of requested results, between 1 and the index size. |
| `search_list` |  | `uint32_t` | Search list size, between k and the index size. |
| `ids` |  | `uint64_t*` | Output buffer of k vector ordinals. |
| `distances` |  | `float*` | Output buffer of k squared distances. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsdiskannclose"></a>
### cuvsDiskAnnClose

Release an owned handle. Zero is accepted; a closed handle must not be reused.

```c
cuvsError_t cuvsDiskAnnClose(cuvsDiskAnnIndex_t handle);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | [`cuvsDiskAnnIndex_t`](/api-reference/c-api-neighbors-diskann#cuvsdiskannindex-t) | Reader handle to release. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsdiskannprepare"></a>
### cuvsDiskAnnPrepare

Convert a cuVS Vamana graph to disk layout and plain PQ.

```c
cuvsError_t cuvsDiskAnnPrepare(const char* data,
const char* graph,
const char* prefix,
uint32_t chunks,
uint32_t sample_rows,
uint32_t block_rows);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `data` |  | `const char*` | Little-endian float32 matrix with uint32 row and dimension headers. |
| `graph` |  | `const char*` | Serialized cuVS Vamana graph for the same vector order. |
| `prefix` |  | `const char*` | New output prefix; its artifact files must not already exist. |
| `chunks` |  | `uint32_t` | PQ chunks, between 1 and the vector dimension. |
| `sample_rows` |  | `uint32_t` | Maximum PQ training rows, between 256 and 256000. |
| `block_rows` |  | `uint32_t` | Encoding block rows, between 1 and 256000. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
