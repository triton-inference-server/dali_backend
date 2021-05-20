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

#ifndef TRITONDALIBACKEND_IO_BUFFER_H
#define TRITONDALIBACKEND_IO_BUFFER_H

#include "src/dali_executor/io_descriptor.h"
#include "src/dali_executor/utils/dali.h"

namespace triton { namespace backend { namespace dali {


void MemCopy(device_type_t dst_dev, void *dst, device_type_t src_dev, const void *src, size_t size,
             cudaStream_t stream = 0);

class IOBufferI {
 public:
  /**
   * @brief Resize the buffer to a given szie.
   * @param size New size.
   */
  virtual void resize(size_t size) = 0;

  /**
   * @brief Get device type of the allocated memory.
   * @return Device type.
   */
  virtual device_type_t device_type() const = 0;

  /**
   * @brief Get an immutable descriptor of the buffer.
   * @return Input buffer descriptor.
   */
  virtual IBufferDescr get_descr() const = 0;

  /**
   * @brief Get a mutable descriptor of the buffer.
   * @return Output buffer descriptor.
   */
  virtual OBufferDescr get_descr() = 0;

  virtual ~IOBufferI() {}

 protected:
  IOBufferI() {}
};

template<typename T, device_type_t Dev>
using buffer_t = std::conditional_t<Dev == device_type_t::CPU, std::vector<T>, DeviceBuffer<T>>;

template<device_type_t Dev>
class IOBuffer : public IOBufferI {
 public:
  explicit IOBuffer(size_t size = 0) : buffer_() {
    resize(size);
    if (Dev == device_type_t::GPU) {
      CUDA_CALL_GUARD(cudaGetDevice(&device_id_));
    }
  }

  void resize(size_t size) override {
    buffer_.resize(size);
  }

  device_type_t device_type() const override {
    return Dev;
  }

  IBufferDescr get_descr() const override {
    IBufferDescr descr;
    descr.data = buffer_.data();
    descr.size = buffer_.size();
    descr.device = Dev;
    descr.device_id = device_id_;
    return descr;
  }

  OBufferDescr get_descr() override {
    OBufferDescr descr;
    descr.data = buffer_.data();
    descr.size = buffer_.size();
    descr.device = Dev;
    descr.device_id = device_id_;
    return descr;
  }

 private:
  buffer_t<uint8_t, Dev> buffer_;
  int device_id_ = 0;
};

}}}  // namespace triton::backend::dali

#endif  // TRITONDALIBACKEND_IO_BUFFER_H
