// The MIT License (MIT)
//
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
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

#ifndef DALI_BACKEND_DALI_EXECUTOR_UTILS_DALI_TYPES_H_
#define DALI_BACKEND_DALI_EXECUTOR_UTILS_DALI_TYPES_H_

#include <dali/core/dali_data_type.h>

#include <cstdint>

/**
 * Types shared by the whole backend to describe data exchanged with DALI.
 *
 * This header must not include any of the DALI C API headers (dali/c_api.h, dali/dali.h).
 * The C API is used only in dali_pipeline.cc, which converts these types at the boundary.
 */

namespace triton::backend::dali {

/**
 * @brief The kind of memory a buffer is located in.
 */
enum class StorageDevice : int {
  CPU = 0,
  GPU = 1,
};

/**
 * @brief Size, in bytes, of a single element of the given type.
 */
inline int64_t dali_type_size(daliDataType_t type) {
  if (type == DALI_BOOL || type == DALI_UINT8 || type == DALI_INT8)
    return 1;
  if (type == DALI_UINT16 || type == DALI_INT16 || type == DALI_FLOAT16)
    return 2;
  if (type == DALI_UINT32 || type == DALI_INT32 || type == DALI_FLOAT)
    return 4;
  else
    return 8;
}

}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_UTILS_DALI_TYPES_H_
