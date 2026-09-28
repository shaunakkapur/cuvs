/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Single-graph index preparation built from pinned DiskANN primitives.
#include "pq.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
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

}  // namespace

extern "C" cuvsError_t cuvsDiskAnnPrepare(const char* data,
                                          const char* graph,
                                          const char* prefix,
                                          uint32_t chunks,
                                          uint32_t sampleLimit,
                                          uint32_t blockRows)
{
  return cuvs::core::translate_exceptions([&] {
    if (!data || !graph || !prefix) throw std::invalid_argument("Null preparation path");
    auto [count, dimension] = matrix_metadata(data, sizeof(float));
    if (count < 256 || count > 1'000'000 || dimension > 128 || chunks < 1 ||
        static_cast<uint32_t>(chunks) > dimension || sampleLimit < 256 || sampleLimit > 256000 ||
        blockRows < 1 || blockRows > 256000)
      throw std::invalid_argument("Invalid bounded preparation bounds");
    validate_finite_matrix(data, count, dimension);
    uint32_t medoid = validate_graph(graph, count);

    const std::string output_prefix(prefix);
    for (const char* suffix : OUTPUT_SUFFIXES)
      if (std::filesystem::exists(output_prefix + suffix))
        throw std::invalid_argument("Output prefix already exists");
    OutputRollback rollback{output_prefix};

    std::filesystem::path staging = output_prefix + "_prepare";
    if (std::filesystem::exists(staging))
      throw std::invalid_argument("Prepare scratch already exists");
    std::filesystem::create_directories(staging);
    RemoveTree cleanup{staging};

    diskann::create_disk_layout<float>(data, graph, output_prefix + "_disk.index");
    write_medoid(output_prefix + "_disk.index_medoids.bin", medoid);
    generate_bounded_pq(data,
                        output_prefix,
                        staging,
                        count,
                        dimension,
                        static_cast<uint32_t>(chunks),
                        static_cast<uint32_t>(sampleLimit),
                        static_cast<uint32_t>(blockRows));

    rollback.committed = true;
  });
}
