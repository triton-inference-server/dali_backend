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

void DaliExecutor::SetupInputs(const std::vector<IDescr>& inputs) {
  assert(!inputs.empty());
  int batch_size = inputs[0].meta.shape.num_samples();
  for (size_t i = 1; i < inputs.size(); ++i) {
    assert(inputs[i].meta.shape.num_samples() == batch_size &&
           "All inputs should have equal batch size.");
  }
  std::vector<IDescr> c_inputs{};
  for (auto& inp : inputs) {
    size_t inp_size = inp.meta.shape.num_elements() * dali_type_size(inp.meta.type);
    if (IsNoCopy(inp)) {
      assert(inp_size <= inp.buffers[0].size);
      c_inputs.push_back(inp);
    } else {
      // Copy buffers to a contiguous buffer on the proper device
      c_inputs.push_back(ScheduleInputCopy(inp));
      assert(inp_size <= c_inputs.back().buffers[0].size);
    }
  }
  RunInputCopy();
  for (auto& inp : c_inputs) {
    pipeline_.SetInput(inp);
  }
}

IDescr DaliExecutor::ScheduleInputCopy(const IDescr& input) {
  assert(input.buffers.size() > 0);
  IOBufferI* buffer;
  if (input.buffers[0].device == device_type_t::CPU) {
    buffer = &cpu_buffers_[input.meta.name];
  } else {
    buffer = &gpu_buffers_[input.meta.name];
  }
  buffer->Clear();
  size_t size = 0;
  for (auto& buf : input.buffers)
    size += buf.size;
  buffer->Reserve(size);
  for (auto& buf : input.buffers) {
    auto dst = buffer->Allocate(buf.size);
    thread_pool_.AddWork([buffer, dst, buf](int) {
      MemCopy(buffer->DeviceType(), dst, buf.device, buf.data, buf.size);
    });
  }
  return IDescr{input.meta, {buffer->GetDescr()}};
}

void DaliExecutor::RunInputCopy() {
  thread_pool_.RunAll();
}

bool DaliExecutor::IsNoCopy(const IDescr& input) {
  return input.buffers.size() == 1 && (input.buffers[0].device == device_type_t::CPU ||
                                       input.buffers[0].device_id == pipeline_.DeviceId());
}

std::vector<OutputInfo> DaliExecutor::Run(const std::vector<IDescr>& inputs) {
  SetupInputs(inputs);
  try {
    pipeline_.Run();
    pipeline_.Output();
  } catch (std::runtime_error& e) {
    pipeline_.Reset();
    throw e;
  }
  std::vector<OutputInfo> ret(pipeline_.GetNumOutput());
  auto outputs_shapes = pipeline_.GetOutputShapes();
  for (size_t out_idx = 0; out_idx < ret.size(); out_idx++) {
    ret[out_idx] = {outputs_shapes[out_idx], pipeline_.GetOutputType(out_idx),
                    pipeline_.GetOutputDevice(out_idx)};
  }
  return ret;
}

void DaliExecutor::PutOutputs(const std::vector<ODescr>& outputs) {
  for (uint32_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
    ENFORCE(outputs[output_idx].buffers.size() == 1,
            "Ouptut can be copied only to a single buffer");
    auto buffer = outputs[output_idx].buffers[0];
    auto data = buffer.data;
    auto device_type = buffer.device;
    pipeline_.PutOutput(data, output_idx, device_type);
  }
}

}}}  // namespace triton::backend::dali
