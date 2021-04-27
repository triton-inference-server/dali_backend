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

#include "src/dali_executor/io_buffer.h"

namespace triton { namespace backend { namespace dali {

void CopyMem(device_type_t dst_dev, void *dst, device_type_t src_dev, const void *src, size_t size,
             cudaStream_t stream) {
  auto src_c = reinterpret_cast<const char *>(src);
  auto dst_c = reinterpret_cast<char *>(dst);
  if (dst_dev == device_type_t::CPU && src_dev == device_type_t::CPU) {
    copyH2H(dst_c, src_c, size, stream);
  } else if (dst_dev == device_type_t::GPU && src_dev == device_type_t::CPU) {
    copyH2D(dst_c, src_c, size, stream);
  } else if (dst_dev == device_type_t::CPU && src_dev == device_type_t::GPU) {
    copyD2H(dst_c, src_c, size, stream);
  } else if (dst_dev == device_type_t::GPU && src_dev == device_type_t::GPU) {
    copyD2D(dst_c, src_c, size, stream);
  }
}

}}}  // namespace triton::backend::dali