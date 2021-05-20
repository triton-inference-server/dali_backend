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

#include "src/dali_executor/dali_pipeline.h"

#include <memory>

namespace triton { namespace backend { namespace dali {

std::once_flag DaliPipeline::dali_initialized_{};


TensorListShape<> DaliPipeline::GetOutputShapeAt(int output_idx) {
  size_t batch_size = daliNumTensors(&handle_, output_idx);
  auto ndim = daliMaxDimTensors(&handle_, output_idx);
  TensorListShape<> result(batch_size, ndim);
  for (size_t s = 0; s < batch_size; ++s) {
    auto* shape = daliShapeAtSample(&handle_, output_idx, s);
    result.set_tensor_shape(s, span<int64_t>(shape, ndim));
    free(shape);
  }
  return result;
}


std::vector<TensorListShape<>> DaliPipeline::GetOutputShapes() {
  auto n_outputs = GetNumOutput();
  std::vector<TensorListShape<>> ret;
  ret.reserve(n_outputs);
  for (int output_idx = 0; output_idx < n_outputs; output_idx++) {
    ret.push_back(GetOutputShapeAt(output_idx));
  }
  return ret;
}


void DaliPipeline::SetInput(const void* data_ptr, const char* name, device_type_t source_device,
                            dali_data_type_t data_type, span<const int64_t> inputs_shapes,
                            int sample_ndims) {
  ENFORCE(inputs_shapes.size() % sample_ndims == 0, "Incorrect inputs shapes or sample ndims");
  int batch_size = inputs_shapes.size() / sample_ndims;
  daliSetExternalInputBatchSize(&handle_, name, batch_size);
  daliSetExternalInput(&handle_, name, source_device, data_ptr, data_type, inputs_shapes.data(),
                       sample_ndims, nullptr, DALI_ext_default);
}


void DaliPipeline::SetInput(const void* ptr, const char* name, device_type_t source_device,
                            dali_data_type_t data_type, TensorListShape<> input_shape) {
  SetInput(ptr, name, source_device, data_type, make_span(input_shape.shapes),
           input_shape.sample_dim());
}

void DaliPipeline::SetInput(const IDescr& io_descr) {
  ENFORCE(io_descr.buffers.size() == 1, "DALI pipeline input has to be a single chunk of memory");
  auto meta = io_descr.meta;
  auto buffer = io_descr.buffers[0];
  SetInput(buffer.data, meta.name.c_str(), buffer.device, meta.type, meta.shape);
}

void DaliPipeline::SyncOutputStream() {
  if (NoGpu())
    return;
  DeviceGuard dg(device_id_);
  CUDA_CALL_GUARD(cudaStreamSynchronize(output_stream_));
}

void DaliPipeline::PutOutput(void* destination, int output_idx, device_type_t destination_device) {
  assert(destination != nullptr);
  assert(output_idx >= 0);
  daliOutputCopy(&handle_, destination, output_idx, destination_device, output_stream_, 0);
}

}}}  // namespace triton::backend::dali
