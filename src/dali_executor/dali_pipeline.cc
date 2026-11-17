// The MIT License (MIT)
//
// Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES
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

#include "src/dali_executor/dali_pipeline.h"

// This is the only file in the backend that may include the DALI C API.
#include <dali/c_api.h>
#include <dali/operators.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>

namespace triton { namespace backend { namespace dali {

namespace {

std::once_flag dali_initialized;

/**
 * @brief Calls `f`, translating DALI's internal exception type into DaliCApiError.
 *
 * The legacy C API reports errors by letting DALI's C++ exceptions escape from the library.
 * Other exception types pass through unchanged.
 */
template<typename F>
decltype(auto) TranslateErrors(F&& f) {
  try {
    return std::forward<F>(f)();
  } catch (const DALIException& e) {
    throw DaliCApiError(e.what());
  }
}

device_type_t ToCApi(StorageDevice device) {
  return device == StorageDevice::GPU ? device_type_t::GPU : device_type_t::CPU;
}

StorageDevice FromCApi(device_type_t device) {
  return device == device_type_t::GPU ? StorageDevice::GPU : StorageDevice::CPU;
}

void InitDali() {
  std::call_once(dali_initialized, []() {
    daliInitialize();
    daliInitOperators();
  });
}

}  // namespace


struct DaliPipeline::Impl {
  daliPipelineHandle handle = nullptr;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  /**
   * @brief Deletes the pipeline, if there is one.
   *
   * The handle must be cleared here: daliDeletePipeline in DALI 2.4 and older neither clears it
   * nor accepts a null one.
   */
  void Delete() {
    if (!handle)
      return;
    daliDeletePipeline(&handle);
    handle = nullptr;
  }

  /**
   * DaliPipeline::ReleasePipeline deletes the pipeline explicitly, so that errors propagate.
   * This only cleans up when a DaliPipeline constructor fails after creating the pipeline.
   */
  ~Impl() {
    try {
      Delete();
    } catch (...) {
      // Nothing more can be done in a destructor.
    }
  }
};


DaliPipeline::DaliPipeline(const std::string& serialized_pipeline, int max_batch_size,
                           int num_threads, int device_id) :
    serialized_pipeline_(serialized_pipeline),
    max_batch_size_(max_batch_size),
    num_threads_(num_threads),
    device_id_(device_id) {
  TranslateErrors([&] {
    DeviceGuard dg(device_id_);
    InitDali();
    // Create the pipeline before the stream, so that neither leaks when the other one fails.
    CreatePipeline();
    InitStream();
  });
}


DaliPipeline::DaliPipeline(DaliPipeline&& dp) {
  *this = std::move(dp);
}


DaliPipeline& DaliPipeline::operator=(DaliPipeline&& dp) {
  if (this != &dp) {
    ReleasePipeline();
    ReleaseStream();
    serialized_pipeline_ = std::move(dp.serialized_pipeline_);
    max_batch_size_ = dp.max_batch_size_;
    num_threads_ = dp.num_threads_;
    device_id_ = dp.device_id_;
    impl_ = std::move(dp.impl_);
    output_stream_ = dp.output_stream_;

    dp.output_stream_ = nullptr;
  }
  return *this;
}


DaliPipeline::~DaliPipeline() {
  ReleasePipeline();
  ReleaseStream();
}


DaliPipeline::Impl& DaliPipeline::GetImpl() {
  ENFORCE(impl_ && impl_->handle, "The DALI pipeline has not been created.");
  return *impl_;
}


void DaliPipeline::CreatePipeline() {
  auto impl = std::make_unique<Impl>();
  TranslateErrors([&] {
    daliCreatePipeline(&impl->handle, serialized_pipeline_.c_str(), serialized_pipeline_.length(),
                       max_batch_size_, num_threads_, device_id_, 0, 1, 0, 0, 0);
  });
  impl_ = std::move(impl);
}


void DaliPipeline::ReleasePipeline() {
  if (!impl_)
    return;
  TranslateErrors([&] { impl_->Delete(); });
  impl_.reset();
}


void DaliPipeline::ReleaseStream() {
  if (NoGpu())
    return;
  if (output_stream_) {
    CUDA_CALL_GUARD(cudaStreamSynchronize(output_stream_));
    CUDA_CALL_GUARD(cudaStreamDestroy(output_stream_));
    output_stream_ = nullptr;
  }
}


void DaliPipeline::InitStream() {
  if (NoGpu())
    return;
  CUDA_CALL_GUARD(cudaStreamCreate(&output_stream_));
}


void DaliPipeline::Reset() {
  ReleasePipeline();
  CreatePipeline();
}


void DaliPipeline::Run() {
  auto& handle = GetImpl().handle;
  TranslateErrors([&] {
    daliOutputRelease(&handle);
    daliRun(&handle);
  });
}


void DaliPipeline::Output() {
  auto& handle = GetImpl().handle;
  TranslateErrors([&] { daliOutput(&handle); });
}


int DaliPipeline::GetBatchSize() {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return static_cast<int>(daliNumTensors(&handle, 0)); });
}


int DaliPipeline::GetNumOutput() {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return static_cast<int>(daliGetNumOutput(&handle)); });
}


TensorListShape<> DaliPipeline::GetOutputShapeAt(int output_idx) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] {
    size_t batch_size = daliNumTensors(&handle, output_idx);
    auto ndim = daliMaxDimTensors(&handle, output_idx);
    TensorListShape<> result(batch_size, ndim);
    for (size_t s = 0; s < batch_size; ++s) {
      auto* shape = daliShapeAtSample(&handle, output_idx, s);
      result.set_tensor_shape(s, span<int64_t>(shape, ndim));
      free(shape);
    }
    return result;
  });
}


std::vector<TensorListShape<>> DaliPipeline::GetOutputShapes() {
  auto n_outputs = GetNumOutput();
  std::vector<TensorListShape<>> ret;
  ret.reserve(n_outputs);
  for (int output_idx = 0; output_idx < n_outputs; output_idx++) {
    ret.push_back(GetOutputShapeAt(output_idx));
  }
  return ret;
}


size_t DaliPipeline::GetOutputNumElements(int output_idx) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return daliNumElements(&handle, output_idx); });
}


daliDataType_t DaliPipeline::GetOutputType(int output_idx) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return daliTypeAt(&handle, output_idx); });
}


StorageDevice DaliPipeline::GetOutputDevice(int output_idx) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return FromCApi(daliGetOutputDevice(&handle, output_idx)); });
}


void DaliPipeline::SetInput(const void* data_ptr, const char* name, StorageDevice source_device,
                            daliDataType_t data_type, span<const int64_t> inputs_shapes,
                            int sample_ndims, const char* data_id, bool force_no_copy) {
  ENFORCE(inputs_shapes.size() % sample_ndims == 0, "Incorrect inputs shapes or sample ndims");
  int batch_size = inputs_shapes.size() / sample_ndims;
  unsigned int flags = DALI_ext_default;
  if (force_no_copy) {
    flags |= DALI_ext_force_no_copy;
  }
  auto& handle = GetImpl().handle;
  TranslateErrors([&] {
    if (data_id) {
      daliSetExternalInputDataId(&handle, name, data_id);
    }
    const char* layout = daliGetExternalInputLayout(&handle, name);
    daliSetExternalInputBatchSize(&handle, name, batch_size);
    daliSetExternalInput(&handle, name, ToCApi(source_device), data_ptr, data_type,
                         inputs_shapes.data(), sample_ndims, layout, flags);
  });
}


void DaliPipeline::SetInput(const void* ptr, const char* name, StorageDevice source_device,
                            daliDataType_t data_type, TensorListShape<> input_shape,
                            std::optional<std::string_view> data_id, bool force_no_copy) {
  SetInput(ptr, name, source_device, data_type, make_span(input_shape.shapes),
           input_shape.sample_dim(), data_id ? data_id->data() : nullptr, force_no_copy);
}

void DaliPipeline::SetInput(const IDescr& io_descr, std::optional<std::string_view> data_id,
                            bool force_no_copy) {
  ENFORCE(io_descr.buffers.size() == 1, "DALI pipeline input has to be a single chunk of memory");
  auto meta = io_descr.meta;
  auto buffer = io_descr.buffers[0];
  SetInput(buffer.data, meta.name.c_str(), buffer.device, meta.type, meta.shape, data_id,
           force_no_copy);
}

void DaliPipeline::SyncStream() {
  if (NoGpu())
    return;
  DeviceGuard dg(device_id_);
  CUDA_CALL_GUARD(cudaStreamSynchronize(output_stream_));
}

void DaliPipeline::PutOutput(void* destination, int output_idx, StorageDevice destination_device) {
  assert(destination != nullptr);
  assert(output_idx >= 0);
  auto& handle = GetImpl().handle;
  TranslateErrors([&] {
    daliOutputCopy(&handle, destination, output_idx, ToCApi(destination_device), output_stream_, 0);
  });
}

std::vector<std::string> DaliPipeline::ListInputs() {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] {
    int num_inputs = daliGetNumExternalInput(&handle);
    std::vector<std::string> result(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      result[i] = daliGetExternalInputName(&handle, i);
    }
    return result;
  });
}

std::optional<std::vector<int64_t>> DaliPipeline::GetInputShape(const std::string& name) {
  auto& handle = GetImpl().handle;
  int ndim = TranslateErrors([&] { return daliGetExternalInputNdim(&handle, name.c_str()); });
  if (ndim >= 0) {
    return std::vector<int64_t>(ndim, -1);
  } else {
    return std::nullopt;
  }
}

daliDataType_t DaliPipeline::GetInputType(const std::string& name) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return daliGetExternalInputType(&handle, name.c_str()); });
}

StorageDevice DaliPipeline::GetInputDevice(const std::string& name) {
  auto& handle = GetImpl().handle;
  auto backend = TranslateErrors([&] { return daliGetOperatorBackend(&handle, name.c_str()); });
  return (backend == DALI_BACKEND_GPU) ? StorageDevice::GPU : StorageDevice::CPU;
}


std::string DaliPipeline::GetOutputName(int id) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return std::string(daliGetOutputName(&handle, id)); });
}

std::optional<std::vector<int64_t>> DaliPipeline::GetDeclaredOutputShape(int id) {
  auto& handle = GetImpl().handle;
  int ndim = TranslateErrors([&] { return daliGetDeclaredOutputNdim(&handle, id); });
  if (ndim >= 0) {
    return std::vector<int64_t>(ndim, -1);
  } else {
    return std::nullopt;
  }
}

daliDataType_t DaliPipeline::GetDeclaredOutputType(int id) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return daliGetDeclaredOutputDtype(&handle, id); });
}

int DaliPipeline::GetMaxBatchSize() {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&] { return daliGetMaxBatchSize(&handle); });
}

std::optional<std::string> DaliPipeline::TryGetOperatorTrace(std::string_view operator_name,
                                                             std::string_view trace_name) {
  auto& handle = GetImpl().handle;
  return TranslateErrors([&]() -> std::optional<std::string> {
    if (daliHasOperatorTrace(&handle, operator_name.data(), trace_name.data())) {
      return daliGetOperatorTrace(&handle, operator_name.data(), trace_name.data());
    } else {
      return std::nullopt;
    }
  });
}


void DaliPipeline::LoadPluginLibs(const std::vector<std::string>& plugin_paths) {
  try {
    InitDali();
    for (const auto& path : plugin_paths) {
      daliLoadLibrary(path.c_str());
    }
  } catch (const std::exception& e) {
    throw DaliBackendException(e.what());
  } catch (...) {
    throw DaliBackendException("Unknown error");
  }
}


bool DaliPipeline::ValidateDaliPipeline(const std::string& serialized_pipeline) {
  return 0 == TranslateErrors([&] {
           return daliIsDeserializable(serialized_pipeline.c_str(), serialized_pipeline.length());
         });
}


void DaliPipeline::ReleaseUnusedMemory() {
  TranslateErrors([] { daliReleaseUnusedMemory(); });
}

}}}  // namespace triton::backend::dali
