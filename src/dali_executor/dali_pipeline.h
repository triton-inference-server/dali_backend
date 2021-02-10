// The MIT License (MIT)
//
// Copyright (c) 2020 NVIDIA CORPORATION
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

#include "src/dali_executor/utils/dali.h"
#include "src/dali_executor/utils/utils.h"
#include "src/error_handling.h"

using std::cout;
using std::endl;

namespace triton { namespace backend { namespace dali {

class DaliPipeline {
 public:
  explicit DaliPipeline(int max_batch_size = 0) : max_batch_size(max_batch_size)
  {
    std::call_once(dali_initialized_, []() {
      daliInitialize();
      daliInitOperators();
    });
    CUDA_CALL(cudaStreamCreate(&output_stream_));
  }


  DaliPipeline(const DaliPipeline&) = delete;

  DaliPipeline& operator=(const DaliPipeline&) = delete;


  DaliPipeline(DaliPipeline&& dp)
      : max_batch_size(dp.max_batch_size), handle_(dp.handle_),
        output_stream_(dp.output_stream_)
  {
    dp.handle_ = daliPipelineHandle{};
    dp.output_stream_ = nullptr;
  }

  ~DaliPipeline()
  {
    ReleasePipeline();
    ReleaseStream();
  }

  DaliPipeline(
      const std::string& serialized_pipeline, int max_batch_size,
      int device_id = -1, int bytes_per_sample_hint = 0, int num_threads = -1,
      int seed = -1)
      : DaliPipeline(max_batch_size)
  {
    daliCreatePipeline(
        &handle_, serialized_pipeline.c_str(), serialized_pipeline.length(),
        max_batch_size, num_threads, device_id, 0, 1, 666, 666, 0);
    assert(handle_.pipe != nullptr && handle_.ws != nullptr);
  }

  void Run()
  {
    daliOutputRelease(&handle_);
    daliRun(&handle_);
  }

  void Output() { daliOutput(&handle_); }

  int GetBatchSize() { return static_cast<int>(daliNumTensors(&handle_, 0)); }

  int GetNumOutput() { return static_cast<int>(daliGetNumOutput(&handle_)); }

  TensorListShape<> GetOutputShapeAt(int output_idx);

  size_t GetOutputNumElements(int output_idx)
  {
    return daliNumElements(&handle_, output_idx);
  }

  dali_data_type_t GetOutputType(int output_idx)
  {
    return daliTypeAt(&handle_, output_idx);
  }

  std::vector<TensorListShape<>> GetOutputShapes();

  void SetInput(
      const void* data_ptr, const char* name, device_type_t source_device,
      dali_data_type_t data_type, span<const int64_t> inputs_shapes,
      int sample_ndims);

  void SetInput(
      const void* ptr, const char* name, device_type_t source_device,
      dali_data_type_t data_type, TensorListShape<> input_shape);

  void PutOutput(
      void* destination, int output_idx, device_type_t destination_device);

  const int max_batch_size;

 private:
  void ReleasePipeline()
  {
    if (handle_.pipe && handle_.ws) {
      daliDeletePipeline(&handle_);
    }
  }

  void ReleaseStream()
  {
    if (output_stream_) {
      CUDA_CALL(cudaStreamSynchronize(output_stream_));
      CUDA_CALL(cudaStreamDestroy(output_stream_));
      output_stream_ = nullptr;
    }
  }


  daliPipelineHandle handle_{};
  ::cudaStream_t output_stream_ = nullptr;
  static std::once_flag dali_initialized_;
};


}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_DALI_PIPELINE_H_
