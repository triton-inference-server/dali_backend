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

#include "src/dali_executor/io_buffer.h"

namespace triton { namespace backend { namespace dali {

void DaliExecutor::SetupInputs(const std::vector<IODescr>& inputs)
{
  assert(!inputs.empty());
  int batch_size = inputs[0].meta.shape.num_samples();
  for (size_t i = 1; i < inputs.size(); ++i) {
    assert(
        inputs[i].meta.shape.num_samples() == batch_size &&
        "All inputs should have equal batch size.");
  }
  for (auto& inp : inputs) {
    assert(
        inp.meta.shape.num_elements() * dali_type_size(inp.meta.type) <=
        inp.buffer.data.size());
    pipeline_.SetInput(
        inp.buffer.data(), inp.name.c_str(), inp.device, inp.type, inp.shape);
  }
}


template <bool owns>
std::vector<shape_and_type_t>
DaliExecutor::Run(const std::vector<IODescr<owns>>& inputs)
{
  SetupInputs(inputs);
  try {
    pipeline_.Run();
    pipeline_.Output();
  }
  catch (std::runtime_error& e) {
    pipeline_.Reset();
    throw e;
  }
  std::vector<shape_and_type_t> ret(pipeline_.GetNumOutput());
  auto outputs_shapes = pipeline_.GetOutputShapes();
  for (size_t out_idx = 0; out_idx < ret.size(); out_idx++) {
    ret[out_idx] = {outputs_shapes[out_idx], pipeline_.GetOutputType(out_idx)};
  }
  return ret;
}


template <bool owns>
void
DaliExecutor::PutOutputs(const std::vector<IODescr<owns>>& outputs)
{
  for (uint32_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
    auto data = outputs[output_idx].buffer.data();
    auto device_type = outputs[output_idx].device;
    pipeline_.PutOutput(data, output_idx, device_type);
  }
}


// Handful of explicit instantiations to make the development less painful
template std::vector<shape_and_type_t> DaliExecutor::Run(const std::vector<IODescr<true>>&);
template std::vector<shape_and_type_t> DaliExecutor::Run(const std::vector<IODescr<false>>&);
template void DaliExecutor::PutOutputs(const std::vector<IODescr<true>>&);
template void DaliExecutor::PutOutputs(const std::vector<IODescr<false>>&);

}}}  // namespace triton::backend::dali
