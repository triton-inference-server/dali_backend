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

#include "src/dali_executor/pipeline_group.h"
#include <numeric>
#include "src/dali_executor/utils/dali.h"

namespace triton { namespace backend { namespace dali {

namespace {

std::vector<int64_t>
dali_to_list_shape(const TensorListShape<>& shape)
{
  auto dim = shape.ndim;
  auto batch_size = shape.num_samples();
  std::vector<int64_t> result(dim * batch_size);
  for (int64_t s = 0; s < batch_size; ++s) {
    for (int64_t d = 0; d < dim; ++d) {
      result[s * dim + d] = shape[s][d];
    }
  }
  return result;
}


int
sum_batch_size(const std::vector<DaliPipeline*>& pipelines)
{
  int result = 0;
  for (auto pipeline : pipelines) {
    result += pipeline->batch_size;
  }
  return result;
}

}  // namespace

void
PipelineGroup::Output()
{
  for (auto pipeline : pipelines_) pipeline->Output();
}


void
PipelineGroup::Run()
{
  for (auto pipeline : pipelines_) pipeline->Run();
}


std::vector<TensorListShape<>>
PipelineGroup::GetOutputsShape()
{
  std::vector<output_shapes_t> shapes;
  for (auto pipeline : pipelines_) {
    shapes.push_back(pipeline->GetOutputShapes());
  }
  auto batch_size = sum_batch_size(pipelines_);
  output_shapes_t output_shapes;
  for (const auto& sh : shapes[0]) {
    output_shapes.push_back(TensorListShape<>(batch_size, sh.ndim));
  }
  int sample = 0;
  for (const auto& sh : shapes) {
    for (int s = 0; s < sh[0].num_samples(); ++s, ++sample) {
      for (size_t output_idx = 0; output_idx < sh.size(); ++output_idx) {
        output_shapes[output_idx].set_tensor_shape(sample, sh[output_idx][s]);
      }
    }
  }
  return output_shapes;
}

int
PipelineGroup::GetNumOutputs()
{
  assert([&]() -> bool {
    for (int i = 0; i < pipelines_.size(); i++) {
      if (pipelines_[i]->GetNumOutput() != pipelines_[0]->GetNumOutput())
        return false;
    }
    return true;
  }());
  return pipelines_[0]->GetNumOutput();
}

std::vector<dali_data_type_t>
PipelineGroup::GetOutputsTypes()
{
  assert([&]() -> bool {
    for (int i = 0; i < pipelines_.size(); i++) {
      for (int out = 0; out < GetNumOutputs(); out++) {
        if (pipelines_[i]->GetOutputType(out) !=
            pipelines_[0]->GetOutputType(out))
          return false;
      }
    }
    return true;
  }());
  std::vector<dali_data_type_t> ret(GetNumOutputs());
  for (int out_idx = 0; out_idx < ret.size(); out_idx++) {
    ret[out_idx] = pipelines_[0]->GetOutputType(out_idx);
  }
  return ret;
}


void
PipelineGroup::PutOutput(
    void* destination, int output_idx, device_type_t destination_device)
{
  auto ptr = reinterpret_cast<char*>(destination);
  for (auto pipeline : pipelines_) {
    pipeline->PutOutput(ptr, output_idx, destination_device);
    auto elem_size = dali_type_size(pipeline->GetOutputType(output_idx));
    ptr += pipeline->GetOutputNumElements(output_idx) * elem_size;
  }
}


void
PipelineGroup::SetInput(
    const void* ptr, const char* name, device_type_t source_device,
    dali_data_type_t data_type, TensorListShape<> input_shape)
{
  auto inp_ptr = reinterpret_cast<const char*>(ptr);
  auto shape_list = dali_to_list_shape(input_shape);
  const int n_dim = input_shape.ndim;
  int sample = 0;
  for (auto pipeline : pipelines_) {
    auto batch_size = pipeline->batch_size;
    auto shape_span =
        span<const int64_t>(&shape_list[sample * n_dim], n_dim * batch_size);
    pipeline->SetInput(
        inp_ptr, name, source_device, data_type, shape_span, input_shape.ndim);
    size_t size = 0;
    for (int s = sample; s < sample + batch_size; ++s) {
      size += volume(input_shape.tensor_shape(s));
    }
    sample += batch_size;
    inp_ptr += size * dali_type_size(data_type);
  }
}

}}}  // namespace triton::backend::dali
