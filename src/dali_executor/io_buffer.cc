// The MIT License (MIT)
//
// Copyright (c) 2021-2026 NVIDIA CORPORATION
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

void MemCopy(StorageDevice dst_dev, void *dst, StorageDevice src_dev, const void *src, size_t size,
             cudaStream_t stream) {
  auto src_c = reinterpret_cast<const char *>(src);
  auto dst_c = reinterpret_cast<char *>(dst);
  if (dst_dev == StorageDevice::CPU && src_dev == StorageDevice::CPU) {
    // Even a null CUDA stream denotes device ordering in DALI's copy helpers.
    // Host buffers must also work when no CUDA driver is available.
    copyH2H(dst_c, src_c, size, ::dali::AccessOrder::host());
  } else if (dst_dev == StorageDevice::GPU && src_dev == StorageDevice::CPU) {
    copyH2D(dst_c, src_c, size, stream);
  } else if (dst_dev == StorageDevice::CPU && src_dev == StorageDevice::GPU) {
    copyD2H(dst_c, src_c, size, stream);
  } else if (dst_dev == StorageDevice::GPU && src_dev == StorageDevice::GPU) {
    copyD2D(dst_c, src_c, size, stream);
  }
}

}}}  // namespace triton::backend::dali
