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

#include "src/dali_executor/dali_executor.h"

#include "src/dali_executor/utils/dali.h"

namespace triton { namespace backend { namespace dali {

std::vector<int>
distribute_batch_size(int batch_size)
{
  std::vector<int> result;
  for (int part = 1; batch_size > 0; part *= 2, batch_size /= 2) {
    if (batch_size % 2)
      result.push_back(part);
  }
  return result;
}

template <bool owns>
std::vector<shape_and_type_t>
DaliExecutor::Run(const std::vector<IODescr<owns>>& inputs)
{
  auto pipelines = SetupInputs(inputs);
  pipelines.Run();
  pipelines.Output();
  std::vector<shape_and_type_t> ret(pipelines.GetNumOutputs());
  auto outputs_shapes = pipelines.GetOutputsShape();
  auto outputs_types = pipelines.GetOutputsTypes();
  for (int out_idx = 0; out_idx < ret.size(); out_idx++) {
    ret[out_idx] = {outputs_shapes[out_idx], outputs_types[out_idx]};
  }
  return ret;
}

template <bool owns>
void
DaliExecutor::PutOutputs(const std::vector<IODescr<owns>>& outputs)
{
  auto pipelines =
      pipeline_pool_.Get(serialized_pipeline_, batch_sizes_, device_id_);
  for (uint32_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
    auto data = outputs[output_idx].buffer.data();
    auto device_type = outputs[output_idx].device;
    pipelines.PutOutput(data, output_idx, device_type);
  }
}

template <bool owns>
PipelineGroup
DaliExecutor::SetupInputs(const std::vector<IODescr<owns>>& inputs)
{
  assert(!inputs.empty());
  int batch_size = inputs[0].shape.num_samples();
  for (size_t i = 1; i < inputs.size(); ++i) {
    assert(
        inputs[i].shape.num_samples() == batch_size &&
        "All inputs should have equal batch size.");
  }
  batch_sizes_ = distribute_batch_size(batch_size);
  auto pipelines =
      pipeline_pool_.Get(serialized_pipeline_, batch_sizes_, device_id_);
  for (auto& inp : inputs) {
    assert(
        inp.shape.num_elements() * dali_type_size(inp.type) <=
        inp.buffer.size());
    pipelines.SetInput(
        inp.buffer.data(), inp.name.c_str(), inp.device, inp.type, inp.shape);
  }
  return pipelines;
}


// Handful of explicit instantiations to make the development less painful
template std::vector<shape_and_type_t> DaliExecutor::Run(const std::vector<IODescr<true>>&);
template std::vector<shape_and_type_t> DaliExecutor::Run(const std::vector<IODescr<false>>&);
template void DaliExecutor::PutOutputs(const std::vector<IODescr<true>>&);
template void DaliExecutor::PutOutputs(const std::vector<IODescr<false>>&);
template PipelineGroup DaliExecutor::SetupInputs(const std::vector<IODescr<true>>&);
template PipelineGroup DaliExecutor::SetupInputs(const std::vector<IODescr<false>>&);

}}}  // namespace triton::backend::dali
