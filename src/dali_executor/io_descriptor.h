// The MIT License (MIT)
//
// Copyright (c) 2021 NVIDIA CORPORATION
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

#ifndef TRITONDALIBACKEND_IO_DESCRIPTOR_H
#define TRITONDALIBACKEND_IO_DESCRIPTOR_H

#include "src/dali_executor/utils/dali.h"
#include "src/dali_executor/utils/utils.h"
#include "src/error_handling.h"

namespace triton { namespace backend { namespace dali {

template<typename T>
struct BufferDescr {
  device_type_t device{};
  int device_id = 0;
  T *data = nullptr;
  size_t size = 0;

  template<typename S, typename = std::enable_if_t<std::is_same<std::remove_const_t<T>, S>::value>>
  BufferDescr(BufferDescr<S> other) :
      device(other.device), device_id(other.device_id), data(other.data), size(other.size) {}

  BufferDescr(){};
};

using IBufferDescr = BufferDescr<const void>;
using OBufferDescr = BufferDescr<void>;

struct IOMeta {
  std::string name{};
  dali_data_type_t type{};
  TensorListShape<> shape{};
};

template<typename T>
struct IODescr {
  IOMeta meta{};
  std::vector<BufferDescr<T>> buffers{};
};

using IDescr = IODescr<const void>;
using ODescr = IODescr<void>;

}}}  // namespace triton::backend::dali

#endif  // TRITONDALIBACKEND_IO_DESCRIPTOR_H
