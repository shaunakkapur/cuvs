/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Optional CPU reader for DiskANN disk indexes, plus conversion of a serialized
// cuVS Vamana graph into that disk layout with plain PQ.
#include "../core/exceptions.hpp"
#include "disk_utils.h"
#include "linux_aligned_file_reader.h"
#include "pq.h"
#include "pq_flash_index.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuvs/neighbors/diskann.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
uint64_t matrix_bytes(uint32_t rows, uint32_t columns, size_t value_bytes)
{
  if (rows == 0 || columns == 0 || value_bytes == 0 ||
      columns > (std::numeric_limits<uint64_t>::max() - 8) / value_bytes ||
      rows >
        (std::numeric_limits<uint64_t>::max() - 8) / (static_cast<uint64_t>(columns) * value_bytes))
    throw std::invalid_argument("Invalid matrix dimensions");
  return 8 + static_cast<uint64_t>(rows) * columns * value_bytes;
}

std::pair<uint32_t, uint32_t> matrix_metadata(const std::string& path, size_t value_bytes)
{
  std::ifstream input(path, std::ios::binary);
  input.exceptions(std::ios::failbit | std::ios::badbit);
  uint32_t rows, columns;
  input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
  uint64_t expected = matrix_bytes(rows, columns, value_bytes);
  uint64_t actual   = std::filesystem::file_size(path);
  if (actual != expected) throw std::invalid_argument("Invalid matrix file");
  return {rows, columns};
}

void validate_finite_matrix(const std::string& path, uint32_t rows, uint32_t columns)
{
  std::ifstream input(path, std::ios::binary);
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.seekg(8);
  std::array<float, 4096> values;
  uint64_t remaining = static_cast<uint64_t>(rows) * columns;
  while (remaining > 0) {
    size_t count = static_cast<size_t>((std::min)(remaining, static_cast<uint64_t>(values.size())));
    input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    for (size_t i = 0; i < count; ++i)
      if (!std::isfinite(values[i])) throw std::invalid_argument("Non-finite input vector");
    remaining -= count;
  }
}

void validate_reader_artifacts(const std::string& prefix, uint32_t dimension, uint32_t count)
{
  const std::string disk_path = prefix + "_disk.index";
  std::ifstream disk(disk_path, std::ios::binary);
  disk.exceptions(std::ios::failbit | std::ios::badbit);
  uint32_t metadata_rows, metadata_columns;
  disk.read(reinterpret_cast<char*>(&metadata_rows), sizeof(metadata_rows));
  disk.read(reinterpret_cast<char*>(&metadata_columns), sizeof(metadata_columns));
  std::array<uint64_t, 9> metadata;
  disk.read(reinterpret_cast<char*>(metadata.data()), sizeof(metadata));
  const uint64_t min_node_bytes =
    static_cast<uint64_t>(dimension) * sizeof(float) + 2 * sizeof(uint32_t);
  const uint64_t max_node_bytes =
    static_cast<uint64_t>(dimension) * sizeof(float) + 513 * sizeof(uint32_t);
  const uint64_t node_bytes       = metadata[3];
  const uint64_t nodes_per_sector = metadata[4];
  const uint64_t expected_nodes_per_sector =
    node_bytes == 0 ? 0 : diskann::defaults::SECTOR_LEN / node_bytes;
  const uint64_t sectors =
    expected_nodes_per_sector == 0
      ? 0
      : (static_cast<uint64_t>(count) + expected_nodes_per_sector - 1) / expected_nodes_per_sector;
  const uint64_t expected_raw_bytes = (sectors + 1) * diskann::defaults::SECTOR_LEN;
  const uint64_t actual_disk_bytes  = std::filesystem::file_size(disk_path);
  if (metadata_rows != metadata.size() || metadata_columns != 1 || metadata[0] != count ||
      metadata[1] != dimension || metadata[2] >= count || node_bytes < min_node_bytes ||
      node_bytes > max_node_bytes || node_bytes % sizeof(uint32_t) != 0 ||
      nodes_per_sector != expected_nodes_per_sector || nodes_per_sector == 0 || metadata[5] != 0 ||
      metadata[7] != 0 || metadata[8] != expected_raw_bytes || actual_disk_bytes != expected_raw_bytes)
    throw std::invalid_argument("Invalid disk index metadata");

  auto [compressed_rows, chunks] = matrix_metadata(prefix + "_pq_compressed.bin", sizeof(uint8_t));
  if (compressed_rows != count || chunks == 0 || chunks > dimension || chunks > 512)
    throw std::invalid_argument("Invalid compressed PQ metadata");

  std::ifstream pivots(prefix + "_pq_pivots.bin", std::ios::binary);
  pivots.exceptions(std::ios::failbit | std::ios::badbit);
  pivots.seekg(4096);
  uint32_t centers, pivot_dimension;
  pivots.read(reinterpret_cast<char*>(&centers), sizeof(centers));
  pivots.read(reinterpret_cast<char*>(&pivot_dimension), sizeof(pivot_dimension));
  if (centers != 256 || pivot_dimension != dimension)
    throw std::invalid_argument("Invalid PQ pivot metadata");

  const std::string medoids_path = disk_path + "_medoids.bin";
  if (std::filesystem::exists(medoids_path)) {
    auto [medoid_rows, medoid_columns] = matrix_metadata(medoids_path, sizeof(uint32_t));
    if (medoid_rows == 0 || medoid_rows > count || medoid_columns != 1)
      throw std::invalid_argument("Invalid medoid metadata");
    std::ifstream medoids(medoids_path, std::ios::binary);
    medoids.exceptions(std::ios::failbit | std::ios::badbit);
    medoids.seekg(8);
    for (uint32_t i = 0; i < medoid_rows; ++i) {
      uint32_t medoid;
      medoids.read(reinterpret_cast<char*>(&medoid), sizeof(medoid));
      if (medoid >= count) throw std::invalid_argument("Invalid medoid metadata");
    }
  }
}

constexpr const char* OUTPUT_SUFFIXES[] = {
  "_disk.index", "_disk.index_medoids.bin", "_pq_pivots.bin", "_pq_compressed.bin"};

struct RemoveTree {
  std::filesystem::path path;
  ~RemoveTree()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

struct OutputRollback {
  std::string prefix;
  bool committed = false;
  ~OutputRollback()
  {
    if (committed) return;
    std::error_code ignored;
    for (const char* suffix : OUTPUT_SUFFIXES)
      std::filesystem::remove(prefix + suffix, ignored);
  }
};

uint32_t validate_graph(const std::string& path, uint32_t rows)
{
  std::ifstream input(path, std::ios::binary);
  input.exceptions(std::ios::failbit | std::ios::badbit);
  uint64_t expected_size, frozen;
  uint32_t width, medoid;
  input.read(reinterpret_cast<char*>(&expected_size), sizeof(expected_size));
  input.read(reinterpret_cast<char*>(&width), sizeof(width));
  input.read(reinterpret_cast<char*>(&medoid), sizeof(medoid));
  input.read(reinterpret_cast<char*>(&frozen), sizeof(frozen));
  if (expected_size != std::filesystem::file_size(path) || width == 0 || width > 512 ||
      medoid >= rows || frozen != 0)
    throw std::invalid_argument("Invalid Vamana graph metadata");
  for (uint32_t row = 0; row < rows; ++row) {
    uint32_t degree;
    input.read(reinterpret_cast<char*>(&degree), sizeof(degree));
    if (degree == 0 || degree > width) throw std::invalid_argument("Invalid Vamana graph degree");
    for (uint32_t edge = 0; edge < degree; ++edge) {
      uint32_t neighbor;
      input.read(reinterpret_cast<char*>(&neighbor), sizeof(neighbor));
      if (neighbor >= rows) throw std::invalid_argument("Invalid Vamana graph neighbor");
    }
  }
  if (input.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("Vamana graph has trailing rows");
  return medoid;
}

void write_training_sample(const std::string& data,
                           const std::filesystem::path& sample,
                           uint32_t source_rows,
                           uint32_t sample_rows,
                           uint32_t columns)
{
  std::ifstream input(data, std::ios::binary);
  std::ofstream output(sample, std::ios::binary);
  input.exceptions(std::ios::failbit | std::ios::badbit);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.write(reinterpret_cast<const char*>(&sample_rows), sizeof(sample_rows));
  output.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
  std::vector<float> row(columns);
  const uint64_t row_bytes = static_cast<uint64_t>(columns) * sizeof(float);
  for (uint32_t i = 0; i < sample_rows; ++i) {
    uint64_t source = static_cast<uint64_t>(i) * (source_rows - 1) / (sample_rows - 1);
    input.seekg(8 + source * row_bytes);
    input.read(reinterpret_cast<char*>(row.data()), row_bytes);
    output.write(reinterpret_cast<const char*>(row.data()), row_bytes);
  }
}

void generate_bounded_pq(const std::string& data,
                         const std::string& output_prefix,
                         const std::filesystem::path& staging,
                         uint32_t count,
                         uint32_t dimension,
                         uint32_t chunks,
                         uint32_t sample_limit,
                         uint32_t block_rows)
{
  uint32_t training_rows       = (std::min)(count, sample_limit);
  std::filesystem::path sample = staging / "pq-training.bin";
  write_training_sample(data, sample, count, training_rows, dimension);
  std::string sample_codes = (staging / "pq-training.codes").string();
  diskann::generate_quantized_data<float>(sample.string(),
                                          output_prefix + "_pq_pivots.bin",
                                          sample_codes,
                                          diskann::Metric::L2,
                                          1.0,
                                          chunks,
                                          false);
  diskann::generate_pq_data_from_pivots<float>(data,
                                               256,
                                               chunks,
                                               output_prefix + "_pq_pivots.bin",
                                               output_prefix + "_pq_compressed.bin",
                                               false,
                                               static_cast<size_t>(block_rows));
}

void write_medoid(const std::string& path, uint32_t medoid)
{
  std::ofstream output(path, std::ios::binary);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  constexpr uint32_t one = 1;
  output.write(reinterpret_cast<const char*>(&one), sizeof(one));
  output.write(reinterpret_cast<const char*>(&one), sizeof(one));
  output.write(reinterpret_cast<const char*>(&medoid), sizeof(medoid));
}

struct Reader {
  // PQFlashIndex retains a reference to this shared_ptr, so declaration order
  // matters.
  std::shared_ptr<AlignedFileReader> file = std::make_shared<LinuxAlignedFileReader>();
  diskann::PQFlashIndex<float> index{file, diskann::Metric::L2};
};
}  // namespace

extern "C" cuvsError_t cuvsDiskAnnOpen(const char* prefix,
                                       uint32_t dimension,
                                       uint32_t count,
                                       cuvsDiskAnnIndex_t* handle)
{
  return cuvs::core::translate_exceptions([&] {
    if (!prefix || !handle || dimension == 0 || dimension > 128 || count < 256 || count > 1000000)
      throw std::invalid_argument("Invalid DiskANN reader bounds");
    *handle = 0;
    validate_reader_artifacts(prefix, dimension, count);
    auto reader = std::make_unique<Reader>();
    if (reader->index.load(1, prefix) != 0 || reader->index.get_data_dim() != dimension ||
        reader->index.get_num_points() != count)
      throw std::invalid_argument("DiskANN index metadata mismatch");
    *handle = reinterpret_cast<cuvsDiskAnnIndex_t>(reader.release());
  });
}

extern "C" cuvsError_t cuvsDiskAnnSearch(cuvsDiskAnnIndex_t handle,
                                         const float* query,
                                         uint32_t dimension,
                                         uint32_t k,
                                         uint32_t search_list,
                                         uint64_t* ids,
                                         float* distances)
{
  return cuvs::core::translate_exceptions([&] {
    auto* reader = reinterpret_cast<Reader*>(handle);
    if (!reader || !query || !ids || !distances || dimension != reader->index.get_data_dim() ||
        k == 0 || k > reader->index.get_num_points() || search_list < k ||
        search_list > reader->index.get_num_points())
      throw std::invalid_argument("Invalid DiskANN search parameters");
    for (uint32_t i = 0; i < dimension; ++i)
      if (!std::isfinite(query[i])) throw std::invalid_argument("Non-finite query");
    reader->index.cached_beam_search(query, k, search_list, ids, distances, 1);
    std::unordered_set<uint64_t> seen;
    for (uint32_t i = 0; i < k; ++i)
      if (ids[i] >= reader->index.get_num_points() || !std::isfinite(distances[i]) ||
          distances[i] < 0 || !seen.insert(ids[i]).second)
        throw std::runtime_error("Invalid DiskANN result");
  });
}

extern "C" cuvsError_t cuvsDiskAnnClose(cuvsDiskAnnIndex_t handle)
{
  return cuvs::core::translate_exceptions([&] { delete reinterpret_cast<Reader*>(handle); });
}

extern "C" cuvsError_t cuvsDiskAnnPrepare(const char* data,
                                          const char* graph,
                                          const char* prefix,
                                          uint32_t chunks,
                                          uint32_t sample_rows,
                                          uint32_t block_rows)
{
  return cuvs::core::translate_exceptions([&] {
    if (!data || !graph || !prefix) throw std::invalid_argument("Null preparation path");
    auto [count, dimension] = matrix_metadata(data, sizeof(float));
    if (count < 256 || count > 1'000'000 || dimension > 128 || chunks < 1 || chunks > dimension ||
        sample_rows < 256 || sample_rows > 256000 || block_rows < 1 || block_rows > 256000)
      throw std::invalid_argument("Invalid preparation bounds");
    validate_finite_matrix(data, count, dimension);
    uint32_t medoid = validate_graph(graph, count);

    const std::string output_prefix(prefix);
    for (const char* suffix : OUTPUT_SUFFIXES)
      if (std::filesystem::exists(output_prefix + suffix))
        throw std::invalid_argument("Output prefix already exists");
    OutputRollback rollback{output_prefix};

    std::filesystem::path staging = output_prefix + "_prepare";
    if (std::filesystem::exists(staging)) throw std::invalid_argument("Prepare scratch already exists");
    std::filesystem::create_directories(staging);
    RemoveTree cleanup{staging};

    diskann::create_disk_layout<float>(data, graph, output_prefix + "_disk.index");
    write_medoid(output_prefix + "_disk.index_medoids.bin", medoid);
    generate_bounded_pq(
      data, output_prefix, staging, count, dimension, chunks, sample_rows, block_rows);
    rollback.committed = true;
  });
}
