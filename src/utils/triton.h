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

#ifndef DALI_BACKEND_UTILS_TRITON_H_
#define DALI_BACKEND_UTILS_TRITON_H_

#include "src/dali_executor/io_descriptor.h"
#include "src/dali_executor/utils/dali.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace dali {


/**
 * @brief Converts TRITONSERVER_DataType to dali_data_type_t
 */
inline dali_data_type_t to_dali(TRITONSERVER_DataType t) {
  assert(t >= 0 && t <= 13);
  // Use the trick, that TRITONSERVER_DataType and dali_data_type_t
  // types have more-or-less the same order, with few exceptions
  if (t == 0)
    return static_cast<dali_data_type_t>(-1);
  if (t == 1)
    return static_cast<dali_data_type_t>(11);
  if (t == 13)
    return static_cast<dali_data_type_t>(0);
  return static_cast<dali_data_type_t>(t - 2);
}

/**
 * @brief Converts TRITONSERVER_MemoryType to DALI device_type_t
 */
inline device_type_t to_dali(TRITONSERVER_MemoryType t) {
  switch (t) {
    case TRITONSERVER_MEMORY_CPU:
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return device_type_t::CPU;
    case TRITONSERVER_MEMORY_GPU:
      return device_type_t::GPU;
    default:
      throw std::invalid_argument("Unknown memory type");
  }
}

/**
 * @brief Converts dali_data_type_t to TRITONSERVER_DataType
 */
inline TRITONSERVER_DataType to_triton(dali_data_type_t t) {
  assert(-1 <= t && t <= 11);
  // Use the trick, that TRITONSERVER_DataType and dali_data_type_t
  // types have more-or-less the same order, with few exceptions
  if (t == -1)
    return static_cast<TRITONSERVER_DataType>(0);
  if (t == 11)
    return static_cast<TRITONSERVER_DataType>(1);
  return static_cast<TRITONSERVER_DataType>(t + 2);
}

/**
 * @brief Converts device_type_t to TRITONSERVER_MemoryType
 */
inline TRITONSERVER_MemoryType to_triton(device_type_t dev) {
  switch (dev) {
    case CPU:
      return TRITONSERVER_MEMORY_CPU_PINNED;
    case GPU:
      return TRITONSERVER_MEMORY_GPU;
    default:
      throw std::invalid_argument("Unknown memory type");
  }
}

class TritonInput {
 public:
  TritonInput(TRITONBACKEND_Input *handle) : handle_(handle) {
    TRITONSERVER_DataType input_datatype;
    const int64_t *input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    const char *name;
    TRITON_CALL_GUARD(TRITONBACKEND_InputProperties(handle_, &name, &input_datatype, &input_shape,
                                                    &input_dims_count, &byte_size_, &buffer_cnt_));
    meta_.name = std::string(name);
    meta_.type = to_dali(input_datatype);
    auto batch_size = input_shape[0];
    auto sample_shape = TensorShape<>(input_shape + 1, input_shape + input_dims_count);
    auto shape = TensorListShape<>::make_uniform(batch_size, sample_shape);
    meta_.shape = shape;
  }

  IOMeta Meta() const {
    return meta_;
  }

  size_t ByteSize() const {
    return byte_size_;
  }

  uint32_t BufferCount() const {
    return buffer_cnt_;
  }

  /**
   * @brief Request an input buffer.
   * @param idx Input index.
   * @param device_type_t Preferred device type.
   * @param device_id Preferred device id.
   * @return Input buffer descriptor.
   */
  IBufferDescr GetBuffer(uint32_t idx, device_type_t device, int device_id) {
    const void *data;
    size_t size;
    TRITONSERVER_MemoryType mem_type = to_triton(device);
    int64_t mem_type_id = device_id;
    TRITON_CALL_GUARD(
        TRITONBACKEND_InputBuffer(handle_, idx, &data, &size, &mem_type, &mem_type_id));
    IBufferDescr descr;
    descr.device = to_dali(mem_type);
    descr.device_id = mem_type_id;
    descr.data = data;
    descr.size = size;
    return descr;
  }

 private:
  TRITONBACKEND_Input *handle_ = nullptr;
  IOMeta meta_{};
  size_t byte_size_ = 0;
  uint32_t buffer_cnt_ = 0;
};

template<class Actual>
class TritonRequestWrapper {
 public:
  /**
   * @brief Fetch the number of inputs provided by the request.
   */
  uint32_t InputCount() const {
    uint32_t input_cnt;
    TRITON_CALL_GUARD(TRITONBACKEND_RequestInputCount(This(), &input_cnt));
    return input_cnt;
  }

  /**
   * @brief Get the input with a given index.
   */
  TritonInput InputByIdx(uint32_t idx) const {
    TRITONBACKEND_Input *input;
    TRITON_CALL_GUARD(TRITONBACKEND_RequestInputByIndex(This(), idx, &input));
    return TritonInput(input);
  }

 private:
  Actual &This() noexcept {
    return static_cast<Actual &>(*this);
  }

  const Actual &This() const noexcept {
    return static_cast<const Actual &>(*this);
  }
};

/** @brief Owning handle for a Triton request. */
class TritonRequest :
    public UniqueHandle<TRITONBACKEND_Request *, TritonRequest>,
    public TritonRequestWrapper<TritonRequest> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(TRITONBACKEND_Request *, TritonRequest)

  static void DestroyHandle(TRITONBACKEND_Request *request) {
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(
                     request, TRITONSERVER_RequestReleaseFlag::TRITONSERVER_REQUEST_RELEASE_ALL),
                 make_string("Failed releasing a request."));
  }
};

/** @brief Non-owning handle for a Triton request. */
class TritonRequestView : public TritonRequestWrapper<TritonRequestView> {
 public:
  TritonRequestView() = default;

  TritonRequestView(TRITONBACKEND_Request *req) : handle_(req) {}

  operator TRITONBACKEND_Request *() const noexcept {
    return handle_;
  }

 private:
  TRITONBACKEND_Request *handle_ = nullptr;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_UTILS_TRITON_H_
