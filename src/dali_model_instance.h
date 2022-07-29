// The MIT License (MIT)
//
// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
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

#ifndef DALI_BACKEND_DALI_MODEL_INSTANCE_H_
#define DALI_BACKEND_DALI_MODEL_INSTANCE_H_

#include "src/dali_executor/dali_executor.h"
#include "src/dali_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace dali {

struct ProcessingMeta {
  TimeInterval compute_interval{};
  int total_batch_size = 0;
};

struct InputsInfo {
  std::vector<IDescr> inputs;
  std::vector<int> reqs_batch_sizes;  // batch size of each request
};

class DaliModelInstance : public ::triton::backend::BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(DaliModel* model_state,
                                    TRITONBACKEND_ModelInstance* triton_model_instance,
                                    DaliModelInstance** state);

  DaliExecutor& GetDaliExecutor() {
    return *dali_executor_;
  }

  const DaliModel& GetDaliModel() const {
    return *dali_model_;
  }

  void Execute(const std::vector<TritonRequest>& requests);

 private:
  DaliModelInstance(DaliModel* model, TRITONBACKEND_ModelInstance* triton_model_instance) :
      BackendModelInstance(model, triton_model_instance), dali_model_(model) {
    auto serialized_pipeline = dali_model_->GetModelProvider().GetModel();
    auto max_batch_size = dali_model_->MaxBatchSize();
    auto num_threads = dali_model_->GetModelParamters().GetNumThreads();
    DaliPipeline pipeline(serialized_pipeline, max_batch_size, num_threads, GetDaliDeviceId());
    dali_executor_ = std::make_unique<DaliExecutor>(std::move(pipeline));
  }

  void ReportStats(TritonRequestView request, TimeInterval exec, TimeInterval compute,
                   bool success) {
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(triton_model_instance_, request,
                                                             success, exec.start, compute.start,
                                                             compute.end, exec.end),
                 "Failed reporting request statistics.");
  }

  void ReportBatchStats(uint32_t total_batch_size, TimeInterval exec, TimeInterval compute) {
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                     triton_model_instance_, total_batch_size, exec.start, compute.start,
                     compute.end, exec.end),
                 "Failed reporting batch statistics.");
  }

  /**
   * @brief Create a response for each request.
   *
   * @return responses vector
   */
  std::vector<TritonResponse> CreateResponses(const std::vector<TritonRequest>& requests);

  /**
   * @brief Run inference for a given \p request and prepare a response.
   * @return computation time interval and total batch size
   */
  ProcessingMeta ProcessRequests(const std::vector<TritonRequest>& requests,
                                 const std::vector<TritonResponse>& responses);

  /**
   * @brief Generate descriptors of inputs provided by given \p requests
   * @return input descriptors and batch size of each request
   */
  InputsInfo GenerateInputs(const std::vector<TritonRequest>& requests);

  int32_t GetDaliDeviceId() {
    return !CudaStream() ? CPU_ONLY_DEVICE_ID : device_id_;
  }

  /**
   * @brief Allocate outputs expected by given \p requests.
   *
   * Lifetime of the created buffer is bound to each of the \p responses
   * @param batch_sizes batch size of each request
   */
  std::vector<ODescr> AllocateOutputs(const std::vector<TritonRequest>& requests,
                                      const std::vector<TritonResponse>& responses,
                                      const std::vector<int>& batch_sizes,
                                      const std::vector<OutputInfo>& outputs_info);

  std::unique_ptr<DaliExecutor> dali_executor_;
  DaliModel* dali_model_;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_MODEL_INSTANCE_H_
