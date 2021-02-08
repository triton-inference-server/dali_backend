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

#ifndef DALI_BACKEND_DALI_BACKEND_H
#define DALI_BACKEND_DALI_BACKEND_H

#include <chrono>
#include <numeric>

#include "src/dali_executor/dali_executor.h"
#include "src/error_handling.h"
#include "src/model_provider/model_provider.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace dali { namespace detail {

/**
 * Converts TRITONSERVER_DataType to dali_data_type_t
 */
dali_data_type_t
to_dali(TRITONSERVER_DataType t)
{
  assert(t >= 0 && t <= 13);
  // Use the trick, that TRITONSERVER_DataType and dali_data_type_t
  // types have more-or-less the same order, with few exceptions
  if (t == 0)
    return static_cast<dali_data_type_t>(-1);
  if (t == 1)
    return static_cast<dali_data_type_t>(11);
  if (t == 13)
    return static_cast<dali_data_type_t>(0);
  return static_cast<dali_data_type_t>(t - 2);
}

/**
 * Converts dali_data_type_t to TRITONSERVER_DataType
 */
TRITONSERVER_DataType
to_triton(dali_data_type_t t)
{
  assert(-1 <= t && t <= 11);
  // Use the trick, that TRITONSERVER_DataType and dali_data_type_t
  // types have more-or-less the same order, with few exceptions
  if (t == -1)
    return static_cast<TRITONSERVER_DataType>(0);
  if (t == 11)
    return static_cast<TRITONSERVER_DataType>(1);
  return static_cast<TRITONSERVER_DataType>(t + 2);
}


device_type_t
to_dali(TRITONSERVER_MemoryType t)
{
  switch (t) {
    case TRITONSERVER_MEMORY_CPU:
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return device_type_t::CPU;
    case TRITONSERVER_MEMORY_GPU:
      return device_type_t::GPU;
    default:
      throw std::invalid_argument("Unknown memory type");
  }
}

inline uint64_t
capture_time()
{
  return std::chrono::steady_clock::now().time_since_epoch().count();
}


std::vector<IODescr<true>>
GenerateInputs(TRITONBACKEND_Request* request)
{
  uint32_t input_cnt;
  TRITON_CALL_GUARD(TRITONBACKEND_RequestInputCount(request, &input_cnt));
  std::vector<IODescr<true>> ret;

  for (size_t input_idx = 0; input_idx < input_cnt; input_idx++) {
    const char* name;
    TRITON_CALL_GUARD(
        TRITONBACKEND_RequestInputName(request, input_idx, &name));
    TRITONBACKEND_Input* input;
    TRITON_CALL_GUARD(
        TRITONBACKEND_RequestInputByIndex(request, input_idx, &input));
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    TRITON_CALL_GUARD(TRITONBACKEND_InputProperties(
        input, nullptr, &input_datatype, &input_shape, &input_dims_count,
        &input_byte_size, &input_buffer_count));

    assert(ret.size() == input_idx);
    ret.emplace_back(input_byte_size);
    auto& input_desc = ret[input_idx];

    input_desc.name = name;
    input_desc.type = to_dali(input_datatype);
    auto batch_size = input_shape[0];
    auto sample_shape =
        TensorShape<>(input_shape + 1, input_shape + input_dims_count);
    auto shape = TensorListShape<>::make_uniform(batch_size, sample_shape);
    input_desc.shape = shape;

    const void* buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType buffer_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    int64_t buffer_memory_type_id = 0;

    for (size_t buffer_idx = 0; buffer_idx < input_buffer_count; buffer_idx++) {
      TRITON_CALL_GUARD(TRITONBACKEND_InputBuffer(
          input, buffer_idx, &buffer, &buffer_byte_size, &buffer_memory_type,
          &buffer_memory_type_id));
      if (buffer_idx == 0) {
        input_desc.device = to_dali(buffer_memory_type);
        input_desc.device_id = buffer_memory_type_id;
      } else {
        ENFORCE(
            input_desc.device == to_dali(buffer_memory_type),
            "The entire buffer of an input shall reside on the same device "
            "type");
        ENFORCE(
            input_desc.device_id == buffer_memory_type_id,
            "The entire buffer of an input shall reside on the same device id");
      }
      input_desc.append((const char*)buffer, buffer_byte_size);
    }
  }
  return ret;
}


std::vector<IODescr<false>>
AllocateOutputs(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response* response,
    const std::vector<shape_and_type_t>& shapes_and_types)
{
  uint32_t output_cnt;
  TRITON_CALL_GUARD(TRITONBACKEND_RequestOutputCount(request, &output_cnt));
  assert(output_cnt == shapes_and_types.size());
  std::vector<IODescr<false>> ret(output_cnt);
  for (size_t output_idx = 0; output_idx < output_cnt; output_idx++) {
    auto& output_desc = ret[output_idx];
    auto& snt = shapes_and_types[output_idx];
    const char* name;
    TRITONBACKEND_RequestOutputName(request, output_idx, &name);
    output_desc.name = name;

    auto output_shape = array_shape(snt.shape);
    TRITONBACKEND_Output* triton_output;
    TRITON_CALL_GUARD(TRITONBACKEND_ResponseOutput(
        response, &triton_output, name, to_triton(snt.type),
        output_shape.data(), output_shape.size()));
    void* buffer;
    TRITONSERVER_MemoryType memtype = TRITONSERVER_MEMORY_GPU;
    int64_t memid = 0;
    auto buffer_byte_size = std::accumulate(
                                output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>()) *
                            TRITONSERVER_DataTypeByteSize(to_triton(snt.type));
    TRITON_CALL_GUARD(TRITONBACKEND_OutputBuffer(
        triton_output, &buffer, buffer_byte_size, &memtype, &memid));
    output_desc.device = to_dali(memtype);
    output_desc.buffer =
        make_span(reinterpret_cast<char*>(buffer), buffer_byte_size);
  }
  return ret;
}


struct RequestMeta {
  uint64_t compute_start_ns, compute_end_ns;
  int batch_size;
};


RequestMeta
ProcessRequest(
    TRITONBACKEND_Response* response, TRITONBACKEND_Request* request,
    DaliExecutor& executor)
{
  RequestMeta ret;

  auto dali_inputs = GenerateInputs(request);
  ret.batch_size =
      dali_inputs[0].shape.num_samples();  // Batch size is expected to be the
                                           // same in every input
  ret.compute_start_ns = capture_time();
  auto shapes_and_types = executor.Run(dali_inputs);
  ret.compute_end_ns = capture_time();
  // TODO verify shapes_and_types against what's provided in config.pbtxt
  auto dali_outputs = AllocateOutputs(request, response, shapes_and_types);
  executor.PutOutputs(dali_outputs);
  return ret;
}

}}}}  // namespace triton::backend::dali::detail

#endif  // DALI_BACKEND_DALI_BACKEND_H
