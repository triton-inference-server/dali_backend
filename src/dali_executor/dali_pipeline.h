// The MIT License (MIT)
//
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES
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

#include <mutex>
#include <string>
#include <vector>
#include <optional>

#include "src/dali_executor/io_descriptor.h"
#include "src/dali_executor/utils/dali.h"
#include "src/dali_executor/utils/utils.h"
#include "src/error_handling.h"


namespace triton { namespace backend { namespace dali {

class DaliPipeline {
 public:
  DaliPipeline(const DaliPipeline&) = delete;


  DaliPipeline& operator=(const DaliPipeline&) = delete;


  DaliPipeline(DaliPipeline&& dp) {
    *this = std::move(dp);
  }

  DaliPipeline& operator=(DaliPipeline&& dp) {
    if (this != &dp) {
      ReleasePipeline();
      ReleaseStream();
      serialized_pipeline_ = std::move(dp.serialized_pipeline_);
      max_batch_size_ = dp.max_batch_size_;
      num_threads_ = dp.num_threads_;
      device_id_ = dp.device_id_;
      handle_ = dp.handle_;
      output_stream_ = dp.output_stream_;

      dp.handle_ = nullptr;
      dp.output_stream_ = nullptr;
    }
    return *this;
  }

  ~DaliPipeline() {
    ReleasePipeline();
    ReleaseStream();
    if (release_buffers_on_delete_) {
      ReleaseBuffers();
    }
  }

  DaliPipeline(const std::string& serialized_pipeline, int max_batch_size, int num_threads,
               int device_id, bool release_buffers_on_delete) :
      serialized_pipeline_(serialized_pipeline),
      max_batch_size_(max_batch_size),
      num_threads_(num_threads),
      device_id_(device_id),
      release_buffers_on_delete_(release_buffers_on_delete) {
    DeviceGuard dg(device_id_);
    InitDali();
    InitStream();
    CreatePipeline();
  }

  void Run() {
    daliOutputRelease(&handle_);
    daliRun(&handle_);
  }

  void Output() {
    daliOutput(&handle_);
  }

  int GetBatchSize() {
    return static_cast<int>(daliNumTensors(&handle_, 0));
  }

  int GetNumOutput() {
    return static_cast<int>(daliGetNumOutput(&handle_));
  }

  TensorListShape<> GetOutputShapeAt(int output_idx);

  size_t GetOutputNumElements(int output_idx) {
    return daliNumElements(&handle_, output_idx);
  }

  dali_data_type_t GetOutputType(int output_idx) {
    return daliTypeAt(&handle_, output_idx);
  }

  device_type_t GetOutputDevice(int output_idx) {
    return daliGetOutputDevice(&handle_, output_idx);
  }

  device_type_t GetInputDevice(const std::string& name);

  std::vector<TensorListShape<>> GetOutputShapes();

  void SetInput(const void* data_ptr, const char* name, device_type_t source_device,
                dali_data_type_t data_type, span<const int64_t> inputs_shapes, int sample_ndims,
                const char *data_id, bool force_no_copy = true);

  void SetInput(const void* ptr, const char* name, device_type_t source_device,
                dali_data_type_t data_type, TensorListShape<> input_shape,
                std::optional<std::string_view> data_id = {}, bool force_no_copy = true);

  void SetInput(const IDescr& io_descr, std::optional<std::string_view> data_id = {}, bool force_no_copy = true);

  void PutOutput(void* destination, int output_idx, device_type_t destination_device);

  /**
   * @brief Get list of external inputs names in the pipeline.
   */
  std::vector<std::string> ListInputs();

  /**
   * @brief Get declared expected shape of the input with a given name.
   */
  std::optional<std::vector<int64_t>> GetInputShape(const std::string &name);

  /**
   * @brief Get declared exptect data type of the input with a given name.
   */
  dali_data_type_t GetInputType(const std::string &name);

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
  dali_data_type_t GetDeclaredOutputType(int id);

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

  void Reset() {
    ReleasePipeline();
    CreatePipeline();
  }


  int DeviceId() const {
    return device_id_;
  }


  int NumThreadsArg() const {
    return num_threads_;
  }

  static void LoadPluginLibs(const std::vector<std::string>& plugin_paths) {
    try {
      InitDali();
      for (const auto& path : plugin_paths) {
        daliLoadLibrary(path.c_str());
      }
    } catch (const std::exception& e) {
      throw DaliBackendException(e.what());
    } catch (...) {
      throw DaliBackendException("Unknown error");
    }
  }

  /**
   * Checks, whether the string contains a formally valid DALI pipeline.
   * @return True, if `serialized_pipeline `is a formally valid DALI pipeline.
   */
  static bool ValidateDaliPipeline(const std::string& serialized_pipeline) {
    return 0 == daliIsDeserializable(serialized_pipeline.c_str(), serialized_pipeline.length());
  }


 private:
  /**
   * @return True, if this DALI Pipeline does not have GPU available
   */
  bool NoGpu() const noexcept {
    return device_id_ < 0;
  }

  void CreatePipeline() {
    daliCreatePipeline(&handle_, serialized_pipeline_.c_str(), serialized_pipeline_.length(),
                       max_batch_size_, num_threads_, device_id_, 0, 1, 0, 0, 0);
  }


  void ReleasePipeline() {
    if (!handle_) return;
    daliDeletePipeline(&handle_);
    handle_ = nullptr;
  }

  void ReleaseStream() {
    if (NoGpu())
      return;
    if (output_stream_) {
      CUDA_CALL_GUARD(cudaStreamSynchronize(output_stream_));
      CUDA_CALL_GUARD(cudaStreamDestroy(output_stream_));
      output_stream_ = nullptr;
    }
  }

  void ReleaseBuffers() {
    daliReleaseUnusedMemory();
  }

  static void InitDali() {
    std::call_once(dali_initialized_, []() {
      daliInitialize();
      daliInitOperators();
    });
  }

  void InitStream() {
    if (NoGpu())
      return;
    CUDA_CALL_GUARD(cudaStreamCreate(&output_stream_));
  }

  std::string serialized_pipeline_{};
  int max_batch_size_ = 0;
  int num_threads_ = 0;
  int device_id_ = 0;
  bool release_buffers_on_delete_ = false;

  daliPipelineHandle handle_ = nullptr;
  ::cudaStream_t output_stream_ = nullptr;
  static std::once_flag dali_initialized_;
};


}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_DALI_PIPELINE_H_
