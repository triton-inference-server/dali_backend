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
#include "src/utils/triton.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace dali { namespace detail {

inline uint64_t capture_time() {
  return std::chrono::steady_clock::now().time_since_epoch().count();
}


/**
 * Allocate outputs within Triton Server
 * @param request Request, corresponding to which the Response will be allocated
 * @param response Output argument. This Response's memory will be allocated
 * @param shapes_and_types Shape and type of the output for a given index
 * @param output_order Maps output name to its index in the results from DALI
 * processing.
 * @return A descriptor that wraps Triton output
 */
std::vector<ODescr> AllocateOutputs(TRITONBACKEND_Request* request,
                                    TRITONBACKEND_Response* response,
                                    const std::vector<shape_and_type_t>& shapes_and_types,
                                    const std::unordered_map<std::string, int>& output_order) {
  uint32_t output_cnt;
  TRITON_CALL_GUARD(TRITONBACKEND_RequestOutputCount(request, &output_cnt));
  ENFORCE(shapes_and_types.size() == output_cnt,
          make_string("Number of outputs in the model configuration (", output_cnt,
                      ") does not match to the number of outputs from DALI pipeline (",
                      shapes_and_types.size(), ")"));

  std::vector<ODescr> ret(output_cnt);
  for (size_t i = 0; i < output_cnt; i++) {
    const char* name;
    TRITONBACKEND_RequestOutputName(request, i, &name);
    auto output_idx = output_order.at(std::string(name));

    auto& output_desc = ret[output_idx];
    output_desc.meta.name = name;
    auto& snt = shapes_and_types[output_idx];

    auto output_shape = array_shape(snt.shape);
    TRITONBACKEND_Output* triton_output;
    TRITON_CALL_GUARD(TRITONBACKEND_ResponseOutput(response, &triton_output, name,
                                                   to_triton(snt.type), output_shape.data(),
                                                   output_shape.size()));
    TRITONSERVER_MemoryType memtype = TRITONSERVER_MEMORY_GPU;
    int64_t memid = 0;
    auto t_size = TRITONSERVER_DataTypeByteSize(to_triton(snt.type));
    output_desc.buffer.size = volume(output_shape.begin(), output_shape.end()) * t_size;
    TRITON_CALL_GUARD(TRITONBACKEND_OutputBuffer(triton_output, &output_desc.buffer.data,
                                                 output_desc.buffer.size, &memtype, &memid));
    output_desc.buffer.device = to_dali(memtype);
  }
  return ret;
}


}}}}  // namespace triton::backend::dali::detail

#endif  // DALI_BACKEND_DALI_BACKEND_H
