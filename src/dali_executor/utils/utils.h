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

#ifndef DALI_BACKEND_DALI_EXECUTOR_UTILS_UTILS_H_
#define DALI_BACKEND_DALI_EXECUTOR_UTILS_UTILS_H_

#include <cuda_runtime_api.h>
#include "src/dali_executor/utils/dali.h"
#include "src/error_handling.h"

#include "nvtx3/nvToolsExt.h"

namespace triton { namespace backend { namespace dali {

template<int ndims = -1>
TensorShape<ndims> max(TensorListShape<ndims> tls) {
  TensorShape<ndims> max = tls.tensor_shape(0);
  for (int i = 1; i < tls.num_samples(); i++) {
    max = tls.tensor_shape(i).num_elements() > max.num_elements() ? tls.tensor_shape(i) : max;
  }
  return max;
}


/**
 * Reformats TensorListShape, so that returned array has the form:
 * [ batch_size, max_volume(TensorListShape)... ]
 */
template<int ndims = -1>
std::vector<int64_t> array_shape(TensorListShape<ndims> tls) {
  std::vector<int64_t> ret(tls.sample_dim() + 1);
  auto max_ts = max(tls);
  ret[0] = tls.num_samples();
  for (size_t i = 1; i < ret.size(); i++) {
    ret[i] = max_ts[i - 1];
  }
  return ret;
}

template<typename Container, int Dims = -1>
TensorListShape<Dims> cat_list_shapes(const Container &shapes) {
  if (shapes.empty())
    return TensorListShape<Dims>(0);
  int64_t num_samples = 0;
  int64_t ndims = shapes.begin()->sample_dim();
  for (auto &shape : shapes)
    num_samples += shape.num_samples();
  ENFORCE(Dims == -1 || Dims == ndims,
          make_string("Cannot convert shape with ", ndims, " dimensions to shape with ", Dims,
                      " dimensions."));
  TensorListShape<Dims> result(num_samples, ndims);
  int64_t ti = 0;
  for (auto &shape : shapes) {
    ENFORCE(shape.sample_dim() == ndims, "Cannot concatenate shapes of different dimensionality");
    for (int64_t j = 0; j < shape.num_samples(); ++j) {
      result.set_tensor_shape(ti++, shape.tensor_shape_span(j));
    }
  }
  return result;
}

template<typename Container, int Dims = -1>
std::vector<TensorListShape<Dims>> split_list_shape(const TensorListShape<Dims> &shape,
                                                    const Container &batch_sizes) {
  size_t nresults = batch_sizes.size();
  int64_t nsamples = 0;
  for (auto bs : batch_sizes) {
    nsamples += bs;
  }
  ENFORCE(nsamples == shape.num_samples(),
          make_string("Cannot split a shape list with ", shape.num_samples(),
                      " samples to list shapes of total ", nsamples, " samples."));
  std::vector<TensorListShape<Dims>> result(nresults);
  int ri = 0;
  int ti = 0;
  for (auto bs : batch_sizes) {
    TensorListShape<Dims> res_shape(bs, shape.sample_dim());
    for (int64_t i = 0; i < bs; ++i) {
      res_shape.set_tensor_shape(i, shape.tensor_shape_span(ti++));
    }
    result[ri++] = std::move(res_shape);
  }
  return result;
}

namespace detail {

struct NvtxDomain {
  static nvtxDomainHandle_t Get() {
    static NvtxDomain inst;
    return inst.domain_;
  }

 private:
  NvtxDomain() {
    domain_ = nvtxDomainCreateA("DALI Backend");
  }

  ~NvtxDomain() {
    nvtxDomainDestroy(domain_);
  }

  nvtxDomainHandle_t domain_;
};

}  // namespace detail


// Basic timerange for profiling
struct TimeRange {
  static const uint32_t kNvGreen = 0x76B900;
  static const uint32_t kNavy = 0x22577E;
  static const uint32_t kTeal = 0x95D1CC;
  static const uint32_t kPantyPink = 0xBD8BC3;


  explicit TimeRange(const std::string& name, const uint32_t rgb = kPantyPink) {
    nvtxEventAttributes_t att = {};
    att.version = NVTX_VERSION;
    att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    att.colorType = NVTX_COLOR_ARGB;
    att.color = rgb | 0xff000000;
    att.messageType = NVTX_MESSAGE_TYPE_ASCII;
    att.message.ascii = name.c_str();

    nvtxDomainRangePushEx(detail::NvtxDomain::Get(), &att);
    started = true;
  }


  ~TimeRange() {
    stop();
  }


  void stop() {
    if (started) {
      started = false;
      nvtxDomainRangePop(detail::NvtxDomain::Get());
    }
  }


 private:
  bool started = false;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_UTILS_UTILS_H_
