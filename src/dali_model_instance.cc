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
  DeviceGuard dg(GetDaliDeviceId());
  TimeInterval exec_interval{};
  start_timer_ns(exec_interval);
  auto responses = CreateResponses(requests);
  ProcessingMeta proc_meta{};
  TritonError error{};
  try {
    proc_meta = ProcessRequests(requests, responses);
  } catch (...) { error = ErrorHandler(); }
  for (auto& response : responses) {
    SendResponse(std::move(response), TritonError::Copy(error));
  }
  end_timer_ns(exec_interval);
  for (auto& request : requests) {
    ReportStats(request, exec_interval, proc_meta.compute_interval, !error);
  }
  ReportBatchStats(proc_meta.total_batch_size, exec_interval, proc_meta.compute_interval);
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
    for (uint32_t input_idx = 0; input_idx < input_cnt; ++input_idx) {
      auto input = request.InputByIdx(input_idx);
      auto input_byte_size = input.ByteSize();
      auto input_buffer_count = input.BufferCount();
      auto meta = input.Meta();
      auto& idescr = input_map[meta.name];
      for (uint32_t buffer_idx = 0; buffer_idx < input_buffer_count; ++buffer_idx) {
        auto buffer = input.GetBuffer(buffer_idx, device_type_t::CPU, GetDaliDeviceId());
        idescr.buffers.push_back(buffer);
      }
      if (idescr.meta.shape.num_samples() == 0) {
        idescr.meta = meta;
      } else {
        ENFORCE(idescr.meta.type == meta.type,
                make_string("Mismatched type for input ", idescr.meta.name));
        idescr.meta.shape.append(meta.shape);
      }
      if (input_idx == 0) {
        reqs_batch_sizes[ri] = meta.shape.num_samples();
      } else {
        ENFORCE(meta.shape.num_samples() == reqs_batch_sizes[ri],
                "Each input in a request must have the same batch size.");
      }
    }
  }
  for (const auto& descrs : input_map) {
    inputs.push_back(descrs.second);
  }
  return {inputs, reqs_batch_sizes};
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
  ENFORCE(outputs_info.size() == output_cnt,
          make_string("Number of outputs expected by the requests (", output_cnt,
                      ") does not match the number of outputs from DALI pipeline (",
                      outputs_info.size(), ")."));
  const auto& output_indices = dali_model_->GetOutputOrder();
  ENFORCE(output_cnt == output_indices.size(),
          make_string("Number of outputs exptected by the requests (", output_cnt,
                      ") does not match the number of outputs in the config (",
                      output_indices.size(), ")."));
  std::vector<ODescr> outputs(output_cnt);
  outputs.reserve(output_cnt);
  for (const auto& out_index : output_indices) {
    auto name = out_index.first;
    int output_idx = out_index.second;
    auto shapes = split_list_shape(outputs_info[output_idx].shape, batch_sizes);
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

}}}  // namespace triton::backend::dali
