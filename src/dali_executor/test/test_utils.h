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

template<typename T>
constexpr dali_data_type_t dali_data_type() {
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


/**
 * @brief Generate a test input.
 *
 * @tparam T Data type.
 * @tparam R Data generator type.
 *
 * @param[out] buffers Collection of buffer chunks to be filled with data.
 * @param name Input name.
 * @param shapes Shapes of the data chunks.
 * @param generator Data generator.
 * @return Input descriptor.
 */
template<typename T, typename R>
IDescr RandomInput(std::vector<std::vector<T>>& buffers, const std::string& name,
                   const std::vector<TensorListShape<>>& shapes, const R& generator) {
  std::vector<IBufferDescr> buf_descs;
  buffers = std::vector<std::vector<T>>(shapes.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    auto& buffer = buffers[i];
    auto& shape = shapes[i];
    buffer.clear();
    std::generate_n(std::back_inserter(buffer), shape.num_elements(), generator);
    IBufferDescr buf_dscr;
    buf_dscr.data = buffer.data();
    buf_dscr.size = sizeof(T) * buffer.size();
    buf_dscr.device = device_type_t::CPU;
    buf_descs.push_back(buf_dscr);
  }
  IDescr dscr;
  dscr.meta.name = name;
  dscr.meta.shape = cat_list_shapes(shapes);
  dscr.meta.type = dali_data_type<T>();
  dscr.buffers = buf_descs;
  return dscr;
}

}}}}  // namespace triton::backend::dali::test

#endif  // DALI_BACKEND_DALI_EXECUTOR_TEST_TEST_UTILS_H_
