// The MIT License (MIT)
//
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES
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

#include "src/dali_model_instance.h"

namespace triton { namespace backend { namespace dali {

/**
 * @brief Handle exceptions and translate it to a TritonError.
 *
 * Should be called only in a catch block.
 */
TritonError ErrorHandler() {
  TritonError error{};
  try {
    throw;
  } catch (TritonError& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
    error = std::move(e);
  } catch (DaliBackendException& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
    error = TritonError::Unknown(make_string("DALI Backend error: ", e.what()));
  } catch (DALIException& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
    error = TritonError::Unknown(make_string("DALI error: ", e.what()));
  } catch (std::runtime_error& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
    error = TritonError::Unknown(make_string("Runtime error: ", e.what()));
  } catch (std::exception& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
    error = TritonError::Unknown(make_string("Exception: ", e.what()));
  } catch (...) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("Unknown error"));
    error = TritonError::Unknown("Unknown error");
  }
  return error;
}

TRITONSERVER_Error* DaliModelInstance::Create(DaliModel* model_state,
                                              TRITONBACKEND_ModelInstance* triton_model_instance,
                                              DaliModelInstance** state) {
  TRITONSERVER_Error* error = nullptr;  // success
  const char* instance_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  try {
    *state = new DaliModelInstance(model_state, triton_model_instance);
  } catch (const std::exception& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN,
                                  make_string("DALI Backend error: ", e.what()).c_str());
  }

  return error;  // success
}

void DaliModelInstance::Execute(const std::vector<TritonRequest>& requests) {
  if (dali_model_->Batched()) {
    ExecuteBatched(requests);
  } else {
    ExecuteUnbatched(requests);
  }
}

void DaliModelInstance::ExecuteBatched(const std::vector<TritonRequest>& requests) {
  DeviceGuard dg(GetDaliDeviceId());
  TimeInterval exec_interval{};
  start_timer_ns(exec_interval);
  auto responses = CreateResponses(requests);
  ProcessingMeta proc_meta{};
  TritonError error{};
  try {
    proc_meta = ProcessRequests(requests, responses);
  } catch (...) {
    error = ErrorHandler();
  }
  for (auto& response : responses) {
    SendResponse(std::move(response), true, TritonError::Copy(error));
  }
  end_timer_ns(exec_interval);

  TimeRange tr_rep("[DALI BE] Report statistics", TimeRange::kTeal);
  for (auto& request : requests) {
    ReportStats(request, exec_interval, proc_meta.compute_interval, !error);
  }
  ReportBatchStats(proc_meta.total_batch_size, exec_interval, proc_meta.compute_interval);
  tr_rep.stop();
}

void DaliModelInstance::ExecuteUnbatched(const std::vector<TritonRequest>& requests) {
  DeviceGuard dg(GetDaliDeviceId());
  for (auto& request : requests) {
    TimeInterval exec_interval{};
    start_timer_ns(exec_interval);
    TritonError error{};
    TimeInterval compute_interval;
    try {
      compute_interval = ProcessRequest(request);
    } catch (...) {
      error = ErrorHandler();
      auto response = TritonResponse::New(request);
      SendResponse(std::move(response), true, TritonError::Copy(error));
    }

    end_timer_ns(exec_interval);

    TimeRange tr_rep("[DALI BE] Report statistics", TimeRange::kTeal);
    ReportStats(request, exec_interval, compute_interval, !error);
    tr_rep.stop();
  }
}

std::vector<TritonResponse> DaliModelInstance::CreateResponses(
    const std::vector<TritonRequest>& requests) {
  std::vector<TritonResponse> responses;
  responses.reserve(requests.size());
  for (auto& request : requests) {
    responses.push_back(TritonResponse::New(request));
  }
  return responses;
}


ProcessingMeta DaliModelInstance::ProcessRequests(const std::vector<TritonRequest>& requests,
                                                  const std::vector<TritonResponse>& responses) {
  ProcessingMeta ret{};

  TimeRange tr_gi("[DALI BE] GenerateInputs", TimeRange::kTeal);
  auto inputs_info = GenerateInputs(requests);
  tr_gi.stop();

  TimeRange tr_run("[DALI BE] Run processing", TimeRange::kTeal);
  start_timer_ns(ret.compute_interval);
  auto outputs_info = dali_executor_->Run(inputs_info.inputs);
  end_timer_ns(ret.compute_interval);
  for (auto& bs : inputs_info.reqs_batch_sizes) {
    ret.total_batch_size += bs;
  }
  tr_run.stop();

  TimeRange tr_ao("[DALI BE] AllocateOutputs", TimeRange::kTeal);
  auto dali_outputs =
      AllocateOutputs(requests, responses, inputs_info.reqs_batch_sizes, outputs_info);
  tr_ao.stop();

  TimeRange tr_copy("[DALI BE] Copy results", TimeRange::kTeal);
  dali_executor_->PutOutputs(dali_outputs);
  tr_copy.stop();

  return ret;
}

TimeInterval DaliModelInstance::ProcessRequest(const TritonRequest& request) {
  TimeRange tr_gi("[DALI BE] GenerateInputs", TimeRange::kTeal);
  auto inputs = GenerateInputs(request);
  tr_gi.stop();

  TimeInterval compute_interval;
  start_timer_ns(compute_interval);
  do {
    TimeRange tr_run("[DALI BE] Run processing", TimeRange::kTeal);
    auto outputs_info = dali_executor_->Run(inputs);
    tr_run.stop();

    auto response = TritonResponse::New(request);

    TimeRange tr_ao("[DALI BE] AllocateOutputs", TimeRange::kTeal);
    auto dali_outputs = AllocateOutputs(request, response, outputs_info);
    tr_ao.stop();

    TimeRange tr_copy("[DALI BE] Copy results", TimeRange::kTeal);
    dali_executor_->PutOutputs(dali_outputs);
    tr_copy.stop();

    SendResponse(std::move(response), dali_executor_->InputsConsumed());
  } while (!dali_executor_->InputsConsumed());
  end_timer_ns(compute_interval);
  return compute_interval;
}

InputsInfo DaliModelInstance::GenerateInputs(const std::vector<TritonRequest>& requests) {
  uint32_t input_cnt = requests[0].InputCount();
  std::vector<IDescr> inputs;
  inputs.reserve(input_cnt);
  std::unordered_map<std::string, IDescr> input_map;
  std::vector<int> reqs_batch_sizes(requests.size());
  for (size_t ri = 0; ri < requests.size(); ++ri) {
    auto& request = requests[ri];
    ENFORCE(request.InputCount() == input_cnt,
            "Each request must provide the same number of inputs.");
    auto idescrs = GenerateInputs(request);
    reqs_batch_sizes[ri] = idescrs[0].meta.shape.num_samples();
    if (ri == 0) {
      for (auto& input : idescrs) {
        input_map[input.meta.name] = std::move(input);
      }
    } else {
      for (auto& input : idescrs) {
        auto idescr = input_map.find(input.meta.name);
        ENFORCE(idescr != input_map.end(), "Got unexpected input with name " + input.meta.name);
        idescr->second.append(std::move(input));
      }
    }
  }
  for (const auto& descrs : input_map) {
    inputs.push_back(std::move(descrs.second));
  }
  return {inputs, reqs_batch_sizes};
}

std::vector<IDescr> DaliModelInstance::GenerateInputs(const TritonRequest& request) {
  std::vector<IDescr> inputs(request.InputCount());
  int num_samples = 0;
  for (uint32_t input_idx = 0; input_idx < request.InputCount(); ++input_idx) {
    auto input = request.InputByIdx(input_idx);
    auto input_byte_size = input.ByteSize();
    auto input_buffer_count = input.BufferCount();
    auto meta = input.Meta();
    auto& idescr = inputs[input_idx];
    for (uint32_t buffer_idx = 0; buffer_idx < input_buffer_count; ++buffer_idx) {
      auto buffer = input.GetBuffer(buffer_idx, device_type_t::CPU, GetDaliDeviceId());
      idescr.buffers.push_back(std::move(buffer));
    }
    idescr.meta = std::move(meta);

    if (input_idx == 0) {
      num_samples = meta.shape.num_samples();
    } else {
      ENFORCE(meta.shape.num_samples() == num_samples,
              "Each input in a request must have the same batch size.");
    }
  }
  return inputs;
}

void ValidateRequestedOutputs(const TritonRequest& request,
                              const std::vector<OutputInfo>& outputs_info,
                              const std::unordered_map<std::string, int>& output_order) {
  uint32_t output_cnt = request.OutputCount();
  ENFORCE(outputs_info.size() == output_cnt,
          make_string("Number of outputs expected by the requests (", output_cnt,
                      ") does not match the number of outputs from DALI pipeline (",
                      outputs_info.size(), ")."));
  ENFORCE(output_cnt == output_order.size(),
          make_string("Number of outputs exptected by the requests (", output_cnt,
                      ") does not match the number of outputs in the config (", output_order.size(),
                      ")."));
}


std::vector<ODescr> DaliModelInstance::AllocateOutputs(
    const std::vector<TritonRequest>& requests, const std::vector<TritonResponse>& responses,
    const std::vector<int>& batch_sizes, const std::vector<OutputInfo>& outputs_info) {
  assert(requests.size() > 0);
  assert(requests.size() == responses.size());
  assert(requests.size() == batch_sizes.size());
  uint32_t output_cnt = requests[0].OutputCount();
  for (auto& req : requests) {
    ENFORCE(output_cnt == req.OutputCount(),
            "All of the requests must expect the same number of outputs.");
  }
  auto output_indices = dali_model_->GetOutputOrder();
  ValidateRequestedOutputs(requests[0], outputs_info, output_indices);

  std::vector<ODescr> outputs(output_cnt);
  for (const auto& out_index : output_indices) {
    auto name = out_index.first;
    int output_idx = out_index.second;
    auto shapes = split_list_shape(outputs_info[output_idx].shape, batch_sizes);
    if (dali_model_->IsOutputSplit(name)) {
      for (auto& shape : shapes) {
        shape = split_outer_dim(shape);
      }
    }
    std::vector<OBufferDescr> buffers(requests.size());
    IOMeta out_meta{};
    out_meta.name = name;
    out_meta.type = outputs_info[output_idx].type;
    for (size_t ri = 0; ri < requests.size(); ++ri) {
      out_meta.shape = shapes[ri];
      auto output = responses[ri].GetOutput(out_meta);
      buffers[ri] = output.AllocateBuffer(outputs_info[output_idx].device, GetDaliDeviceId());
    }
    out_meta.shape = outputs_info[output_idx].shape;
    outputs[output_idx] = {out_meta, buffers};
  }
  return outputs;
}

std::vector<ODescr> DaliModelInstance::AllocateOutputs(
    const TritonRequest& request, const TritonResponse& response,
    const std::vector<OutputInfo>& outputs_info) {
  auto output_indices = dali_model_->GetOutputOrder();
  ValidateRequestedOutputs(request, outputs_info, output_indices);
  std::vector<ODescr> outputs(request.OutputCount());
  for (const auto& out_index : output_indices) {
    auto name = out_index.first;
    int output_idx = out_index.second;
    OBufferDescr buffer;
    IOMeta out_meta{};
    out_meta.name = name;
    out_meta.type = outputs_info[output_idx].type;
    out_meta.shape = outputs_info[output_idx].shape;

    auto output = response.GetOutput(out_meta);
    buffer = output.AllocateBuffer(outputs_info[output_idx].device, GetDaliDeviceId());
    out_meta.shape = outputs_info[output_idx].shape;
    outputs[output_idx] = {out_meta, {buffer}};
  }
  return outputs;
}

}}}  // namespace triton::backend::dali
