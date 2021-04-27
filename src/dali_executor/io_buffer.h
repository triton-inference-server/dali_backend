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

#include "src/dali_executor/utils/dali.h"

namespace triton { namespace backend { namespace dali {


void copyMem(device_type_t dst_dev, void *dst, device_type_t src_dev, const void *src, size_t size,
             cudaStream_t stream = 0) {
  if (dst_dev == device_type_t::CPU && src_dev == device_type_t::CPU) {
    copyH2H(dst, src, size, stream);
  } else if (dst_dev == device_type_t::GPU && src_dev == device_type_t::CPU) {
    copyH2D(dst, src, size, stream);
  } else if (dst_dev == device_type_t::CPU && src_dev == device_type_t::GPU) {
    copyD2H(dst, src, size, stream);
  } else if (dst_dev == device_type_t::GPU && src_dev == device_type_t::GPU) {
    copyD2D(dst, src, size, stream);
  }
}

class IOBufferI {
 public:
  /**
   * Extend the buffer by the `size` of bytes and return a pointer
   * to the beginning of the extension.
   */
  virtual uint8_t *Extend(size_t size) = 0;

  virtual void Clear() = 0;

  virtual void Reserve(size_t size) = 0;

  virtual size_t Capacity() const = 0;

  virtual device_type_t DeviceType() const = 0;

  virtual BufferDescr GetDescr() const = 0;

  virtual ~IOBufferI() {}

 protected:
  IOBufferI() {}
};

template<typename T, device_type_t Dev>
using buffer_t = std::conditional_t<Dev == device_type_t::CPU, std::vector<T>, DeviceBuffer<T>>;

template<device_type_t Dev>
class IOBuffer : public IOBufferI {
 public:
  IOBuffer(size_t size = 0) : buffer_(size) {
    if (Dev == device_type_t : GPU) {
      CUDA_CALL_GUARD(cudaGetDevice(&device_id));
    }
  }

  uint8_t *Extend(size_t size) override {
    ENFORCE(filled + size < buffer_.size(), "Cannot extend the buffer beyond its capacity.");
    auto origin = buffer_.data() + filled;
    filled += size;
    return origin;
  }

  void Clear() override {
    filled_ = 0;
  }

  void Reserve(size_t size) override {
    if (size > buffer_.size())
      buffer_.resize(size);
  }

  size_t Capacity() const override {
    return buffer_.size();
  }

  device_type_t DeviceType() const override {
    return Dev;
  }

  BufferDescr GetDescr() const override {
    BufferDescr descr;
    descr.data = span<void>(buffer_.data(), filled_);
    descr.device = Dev;
    descr.device_id = device_id;
    return descr;
  }

 private:
  buffer_t<uint8_t, Dev> buffer_;
  size_t filled_ = 0;
  int device_id = 0;
};

}}}  // namespace triton::backend::dali

#endif  // TRITONDALIBACKEND_IO_BUFFER_H