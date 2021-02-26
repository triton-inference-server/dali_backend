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

#ifndef DALI_BACKEND_DALI_EXECUTOR_TEST_TEST_UTILS_H_
#define DALI_BACKEND_DALI_EXECUTOR_TEST_TEST_UTILS_H_

#include <algorithm>
#include <string>
#include <vector>

#include "src/dali_executor/dali_executor.h"

namespace triton { namespace backend { namespace dali { namespace test {

template <typename T>
constexpr dali_data_type_t
dali_data_type()
{
  if (std::is_same<T, uint8_t>::value) {
    return dali_data_type_t::DALI_UINT8;
  } else if (std::is_same<T, uint16_t>::value) {
    return dali_data_type_t::DALI_UINT16;
  } else if (std::is_same<T, uint32_t>::value) {
    return dali_data_type_t::DALI_UINT32;
  } else if (std::is_same<T, uint64_t>::value) {
    return dali_data_type_t::DALI_UINT64;
  } else if (std::is_same<T, int8_t>::value) {
    return dali_data_type_t::DALI_INT8;
  } else if (std::is_same<T, int16_t>::value) {
    return dali_data_type_t::DALI_INT16;
  } else if (std::is_same<T, int32_t>::value) {
    return dali_data_type_t::DALI_INT32;
  } else if (std::is_same<T, int64_t>::value) {
    return dali_data_type_t::DALI_INT64;
  } else if (std::is_same<T, float>::value) {
    return dali_data_type_t::DALI_FLOAT;
  } else if (std::is_same<T, double>::value) {
    return dali_data_type_t::DALI_FLOAT64;
  } else if (std::is_same<T, bool>::value) {
    return dali_data_type_t::DALI_BOOL;
  } else {
    return dali_data_type_t::DALI_NO_TYPE;
  }
}


template <typename T, typename R>
IODescr<false>
RandomInput(
    std::vector<T>& buffer, const std::string& name, TensorListShape<> shape,
    const R& generator)
{
  buffer.clear();
  std::generate_n(std::back_inserter(buffer), shape.num_elements(), generator);
  IODescr<false> dscr;
  dscr.name = name;
  dscr.shape = shape;
  dscr.type = dali_data_type<T>();
  dscr.device = device_type_t::CPU;
  dscr.buffer = span<char>(
      reinterpret_cast<char*>(buffer.data()), sizeof(T) * buffer.size());
  dscr.device = device_type_t::CPU;
  return dscr;
}

}}}}  // namespace triton::backend::dali::test

#endif  // DALI_BACKEND_DALI_EXECUTOR_TEST_TEST_UTILS_H_
