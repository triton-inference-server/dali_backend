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

#ifndef DALI_BACKEND_UTILS_ERROR_HANDLING_H_
#define DALI_BACKEND_UTILS_ERROR_HANDLING_H_

#include <cassert>
#include <stdexcept>
#include <string>

namespace triton { namespace backend { namespace dali {

struct DaliBackendException : public std::runtime_error {
  explicit DaliBackendException(const std::string& msg) : std::runtime_error(msg) {}
};

#define ENFORCE(predicate, message) \
  if (!(predicate))                 \
  throw DaliBackendException(message)


#define TRITON_CALL_GUARD(call_)                                                                \
  do {                                                                                          \
    auto err = call_;                                                                           \
    if (err) {                                                                                  \
      std::stringstream ss;                                                                     \
      ss << "Error " << TRITONSERVER_ErrorCode(err) << "(" << TRITONSERVER_ErrorCodeString(err) \
         << "): " << TRITONSERVER_ErrorMessage(err);                                            \
      throw DaliBackendException(ss.str());                                                     \
    }                                                                                           \
  } while (false)


inline void CudaResultCheck(cudaError_t err) {
  switch (err) {
    case cudaSuccess:
      return;
    case cudaErrorMemoryAllocation:
    case cudaErrorInvalidValue:
    default:
      throw DaliBackendException(make_string(cudaGetErrorName(cudaGetLastError()), " --> ",
                                             cudaGetErrorString(cudaGetLastError())));
      cudaGetLastError();
  }
}

template<typename T>
void CUDA_CALL_GUARD(T status) {
  CudaResultCheck(status);
}

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_UTILS_ERROR_HANDLING_H_
