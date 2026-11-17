// The MIT License (MIT)
//
// Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef DALI_BACKEND_DALI_EXECUTOR_DALI_PIPELINE_H_
#define DALI_BACKEND_DALI_EXECUTOR_DALI_PIPELINE_H_

#include <cuda_runtime_api.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/dali_executor/io_descriptor.h"
#include "src/dali_executor/utils/dali.h"
#include "src/dali_executor/utils/utils.h"
#include "src/error_handling.h"


namespace triton { namespace backend { namespace dali {

/**
 * @brief A DALI pipeline.
 *
 * This is the only class that calls the DALI C API, and dali_pipeline.cc is the only file that
 * includes the C API header. Errors reported by DALI are thrown as DaliCApiError.
 */
class DaliPipeline {
 public:
  DaliPipeline(const std::string& serialized_pipeline, int max_batch_size, int num_threads,
               int device_id);

  DaliPipeline(const DaliPipeline&) = delete;

  DaliPipeline& operator=(const DaliPipeline&) = delete;

  DaliPipeline(DaliPipeline&& dp);

  DaliPipeline& operator=(DaliPipeline&& dp);

  ~DaliPipeline();

  void Run();

  void Output();

  int GetBatchSize();

  int GetNumOutput();

  TensorListShape<> GetOutputShapeAt(int output_idx);

  size_t GetOutputNumElements(int output_idx);

  daliDataType_t GetOutputType(int output_idx);

  StorageDevice GetOutputDevice(int output_idx);

  StorageDevice GetInputDevice(const std::string& name);

  std::vector<TensorListShape<>> GetOutputShapes();

  void SetInput(const void* data_ptr, const char* name, StorageDevice source_device,
                daliDataType_t data_type, span<const int64_t> inputs_shapes, int sample_ndims,
                const char* data_id, bool force_no_copy = true);

  void SetInput(const void* ptr, const char* name, StorageDevice source_device,
                daliDataType_t data_type, TensorListShape<> input_shape,
                std::optional<std::string_view> data_id = {}, bool force_no_copy = true);

  void SetInput(const IDescr& io_descr, std::optional<std::string_view> data_id = {},
                bool force_no_copy = true);

  void PutOutput(void* destination, int output_idx, StorageDevice destination_device);

  /**
   * @brief Get list of external inputs names in the pipeline.
   */
  std::vector<std::string> ListInputs();

  /**
   * @brief Get declared expected shape of the input with a given name.
   */
  std::optional<std::vector<int64_t>> GetInputShape(const std::string& name);

  /**
   * @brief Get declared exptect data type of the input with a given name.
   */
  daliDataType_t GetInputType(const std::string& name);

  /**
   * @brief Get name of the pipeline output with a given id.
   */
  std::string GetOutputName(int id);

  /**
   * @brief Get shape declared for the pipeline output with a given id.
   */
  std::optional<std::vector<int64_t>> GetDeclaredOutputShape(int id);

  /**
   * @brief Get declared output data type for the pipeline output with a given id.
   */
  daliDataType_t GetDeclaredOutputType(int id);

  /**
   * @brief Get max batch size of the pipeline.
   */
  int GetMaxBatchSize();

  /**
   * @brief Wait for the work scheduled on the copy stream.
   *
   * This should be always called after copying all of the pipeline outputs.
   */
  void SyncStream();

  std::optional<std::string> TryGetOperatorTrace(std::string_view operator_name,
                                                 std::string_view trace_name);

  cudaStream_t CopyStream() {
    return output_stream_;
  }

  void Reset();


  int DeviceId() const {
    return device_id_;
  }


  int NumThreadsArg() const {
    return num_threads_;
  }

  static void LoadPluginLibs(const std::vector<std::string>& plugin_paths);

  /**
   * Checks, whether the string contains a formally valid DALI pipeline.
   * @return True, if `serialized_pipeline `is a formally valid DALI pipeline.
   */
  static bool ValidateDaliPipeline(const std::string& serialized_pipeline);

  /**
   * @brief Release unused memory from DALI's memory pools.
   */
  static void ReleaseUnusedMemory();

 private:
  /**
   * @return True, if this DALI Pipeline does not have GPU available
   */
  bool NoGpu() const noexcept {
    return device_id_ < 0;
  }

  void CreatePipeline();

  void ReleasePipeline();

  void ReleaseStream();

  void InitStream();

  /** Owns the C API pipeline handle. Defined in dali_pipeline.cc. */
  struct Impl;

  /** Returns the live pipeline; throws if there is none (e.g. in a moved-from object). */
  Impl& GetImpl();

  std::string serialized_pipeline_{};
  int max_batch_size_ = 0;
  int num_threads_ = 0;
  int device_id_ = 0;

  std::unique_ptr<Impl> impl_;
  ::cudaStream_t output_stream_ = nullptr;
};


}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_DALI_PIPELINE_H_
