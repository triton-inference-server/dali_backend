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

#ifndef DALI_BACKEND_UTILS_DALI_H_
#define DALI_BACKEND_UTILS_DALI_H_

#include <dali/c_api.h>
#include <dali/core/cuda_stream.h>
#include <dali/core/dev_buffer.h>
#include <dali/core/device_guard.h>
#include <dali/core/format.h>
#include <dali/core/span.h>
#include <dali/core/tensor_shape.h>
#include <dali/core/tensor_shape_print.h>
#include <dali/core/unique_handle.h>
#include <dali/core/util.h>
#include <dali/operators.h>
#include <dali/pipeline/util/thread_pool.h>

namespace triton { namespace backend { namespace dali {

using ::dali::copyD2D;
using ::dali::copyD2H;
using ::dali::copyH2D;
using ::dali::copyH2H;
using ::dali::CUDAStream;
using ::dali::DALIException;
using ::dali::DeviceBuffer;
using ::dali::DeviceGuard;
using ::dali::make_cspan;
using ::dali::make_span;
using ::dali::make_string;
using ::dali::span;
using ::dali::TensorListShape;
using ::dali::TensorShape;
using ::dali::ThreadPool;
using ::dali::UniqueHandle;
using ::dali::volume;
using ::dali::CPU_ONLY_DEVICE_ID;


inline int64_t dali_type_size(dali_data_type_t type) {
  if (type == DALI_BOOL || type == DALI_UINT8 || type == DALI_INT8)
    return 1;
  if (type == DALI_UINT16 || type == DALI_INT16 || type == DALI_FLOAT16)
    return 2;
  if (type == DALI_UINT32 || type == DALI_INT32 || type == DALI_FLOAT)
    return 4;
  else
    return 8;
}

}}}  // namespace triton::backend::dali


#endif  // DALI_BACKEND_UTILS_DALI_H_
