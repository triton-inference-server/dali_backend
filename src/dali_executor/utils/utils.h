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

#ifndef DALI_BACKEND_UTILS_UTILS_H_
#define DALI_BACKEND_UTILS_UTILS_H_

#include <cuda_runtime_api.h>
#include "nvtx3/nvToolsExt.h"

namespace triton { namespace backend { namespace dali {

template <int ndims = -1>
TensorShape<ndims>
max(TensorListShape<ndims> tls)
{
  TensorShape<ndims> max = tls.tensor_shape(0);
  for (int i = 1; i < tls.num_samples(); i++) {
    max = tls.tensor_shape(i).num_elements() > max.num_elements()
              ? tls.tensor_shape(i)
              : max;
  }
  return max;
}


/**
 * Reformats TensorListShape, so that returned array has the form:
 * [ batch_size, max_volume(TensorListShape)... ]
 */
template <int ndims = -1>
std::vector<int64_t>
array_shape(TensorListShape<ndims> tls)
{
  std::vector<int64_t> ret(tls.sample_dim() + 1);
  auto max_ts = max(tls);
  ret[0] = tls.num_samples();
  for (size_t i = 1; i < ret.size(); i++) {
    ret[i] = max_ts[i - 1];
  }
  return ret;
}


// Basic timerange for profiling
struct TimeRange {
  static const uint32_t kRed = 0xFF0000;
  static const uint32_t kGreen = 0x00FF00;
  static const uint32_t kBlue = 0x0000FF;
  static const uint32_t kYellow = 0xB58900;
  static const uint32_t kOrange = 0xCB4B16;
  static const uint32_t kRed1 = 0xDC322F;
  static const uint32_t kMagenta = 0xD33682;
  static const uint32_t kViolet = 0x6C71C4;
  static const uint32_t kBlue1 = 0x268BD2;
  static const uint32_t kCyan = 0x2AA198;
  static const uint32_t kGreen1 = 0x859900;
  static const uint32_t knvGreen = 0x76B900;
  static const uint32_t kPantyPink = 0xBD8BC3;


  TimeRange(std::string name, const uint32_t rgb = kPantyPink)
  {  // NOLINT
    nvtxEventAttributes_t att = {};
    att.version = NVTX_VERSION;
    att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    att.colorType = NVTX_COLOR_ARGB;
    att.color = rgb | 0xff000000;
    att.messageType = NVTX_MESSAGE_TYPE_ASCII;
    att.message.ascii = name.c_str();

    nvtxRangePushEx(&att);
    started = true;
  }


  ~TimeRange() { stop(); }


  void stop()
  {
    if (started) {
      started = false;
      nvtxRangePop();
    }
  }


 private:
  bool started = false;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_UTILS_UTILS_H_
