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

#include <memory>

#include "src/dali_executor/dali_executor.h"
#include "src/dali_executor/io_buffer.h"
#include "src/dali_executor/utils/dali.h"
#include "src/dali_executor/utils/utils.h"
#include "src/model_provider/model_provider.h"
#include "src/utils/timing.h"
#include "src/utils/triton.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace dali {

struct ModelParameters {
  explicit ModelParameters(common::TritonJson::Value& model_config) {
    model_config.MemberAsObject("parameters", &params_);
  }

  /**
   * Rerurn a value of a parameter with a given `key`
   * or `def` if the parameter is not present.
   */
  template<typename T>
  T GetParam(const std::string& key, const T& def = T()) {
    T result = def;
    GetMember(key, result);
    return result;
  }

  int GetNumThreads() {
    return GetParam("num_threads", -1);
  }

 private:
  template<typename T>
  void GetMember(const std::string& key, T& value) {
    auto key_c = key.c_str();
    if (params_.Find(key_c)) {
      common::TritonJson::Value param;
      TRITON_CALL_GUARD(params_.MemberAsObject(key_c, &param));
      std::string string_value;
      TRITON_CALL_GUARD(param.MemberAsString("string_value", &string_value));
      value = from_string<T>(string_value);
    }
  }

  common::TritonJson::Value params_;
};

class DaliModel : public ::triton::backend::BackendModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, DaliModel** state);

  virtual ~DaliModel() = default;

  TRITONSERVER_Error* ValidateModelConfig() {
    // We have the json DOM for the model configuration...
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("model configuration:\n") + buffer.Contents()).c_str());

    return nullptr;  // success
  }

  const ModelProvider& GetModelProvider() const {
    return *dali_model_provider_;
  };

  ModelParameters& GetModelParamters() {
    return params_;
  }


  void ReadOutputsOrder() {
    using Value = ::triton::common::TritonJson::Value;
    Value outputs;
    model_config_.MemberAsArray("output", &outputs);
    for (size_t output_idx = 0; output_idx < outputs.ArraySize(); output_idx++) {
      Value out;
      std::string name;
      outputs.IndexAsObject(output_idx, &out);
      out.MemberAsString("name", &name);
      output_order_[name] = output_idx;
    }
  }


  const std::unordered_map<std::string, int>& GetOutputOrder() const {
    return output_order_;
  }


 private:
  explicit DaliModel(TRITONBACKEND_Model* triton_model) :
      BackendModel(triton_model), params_(model_config_) {
    const char sep = '/';

    const char* model_repo_path;
    TRITONBACKEND_ArtifactType artifact_type;
    TRITON_CALL_GUARD(
        TRITONBACKEND_ModelRepository(triton_model_, &artifact_type, &model_repo_path));

    std::stringstream dali_pipeline_path;
    dali_pipeline_path << model_repo_path << sep << version_ << sep << GetModelFilename();
    std::string filename = dali_pipeline_path.str();
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (make_string("Loading DALI pipeline from file ", filename).c_str()));
    dali_model_provider_ = std::make_unique<FileModelProvider>(filename);
  }

  std::string GetModelFilename() {
    std::string ret;
    TRITON_CALL_GUARD(model_config_.MemberAsString("default_model_filename", &ret));
    return ret.empty() ? "model.dali" : ret;
  }

  ModelParameters params_;
  std::unique_ptr<ModelProvider> dali_model_provider_;
  std::unordered_map<std::string, int> output_order_;
};


TRITONSERVER_Error* DaliModel::Create(TRITONBACKEND_Model* triton_model, DaliModel** state) {
  TRITONSERVER_Error* error = nullptr;  // success
  try {
    *state = new DaliModel(triton_model);
  } catch (const std::exception& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN,
                                  make_string("DALI Backend error: ", e.what()).c_str());
  }

  return error;
}

struct RequestMeta {
  TimeInt compute_interval;
  int batch_size;
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

  void Execute(std::vector<TritonRequest> requests) {
    DeviceGuard dg(device_id_);
    int total_batch_size = 0;
    TimeInt batch_compute_interval{};
    Timer batch_exec_timer{};
    for (size_t i = 0; i < requests.size(); i++) {
      auto response = TritonResponse::New(requests[i]);
      RequestMeta request_meta;
      TritonError error{};
      TimeInt req_exec_interval;
      try {
        Timer req_timer{};
        request_meta = ProcessRequest(response, requests[i]);
        req_exec_interval = req_timer.Interval();
      } catch (...) { error = ErrorHandler(); }

      if (i == 0) {
        batch_compute_interval.start = request_meta.compute_interval.start;
      } else if (i == requests.size() - 1) {
        batch_compute_interval.end = request_meta.compute_interval.end;
      }

      ReportStats(requests[i], req_exec_interval, request_meta.compute_interval, !error);
      SendResponse(std::move(response), std::move(error));

      total_batch_size += request_meta.batch_size;
    }
    TimeInt batch_exec_interval = batch_exec_timer.Interval();
    ReportBatchStats(total_batch_size, batch_exec_interval, batch_compute_interval);
  }

 private:
  DaliModelInstance(DaliModel* model, TRITONBACKEND_ModelInstance* triton_model_instance) :
      BackendModelInstance(model, triton_model_instance), dali_model_(model) {
    auto serialized_pipeline = dali_model_->GetModelProvider().GetModel();
    auto max_batch_size = dali_model_->MaxBatchSize();
    auto num_threads = dali_model_->GetModelParamters().GetNumThreads();
    DaliPipeline pipeline(serialized_pipeline, max_batch_size, num_threads, device_id_);
    dali_executor_ = std::make_unique<DaliExecutor>(std::move(pipeline));
  }

  void ReportStats(TritonRequestView request, TimeInt exec, TimeInt compute, bool success) {
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(triton_model_instance_, request,
                                                             success, exec.start, compute.start,
                                                             compute.end, exec.end),
                 "Failed reporting request statistics.");
  }

  void ReportBatchStats(uint32_t total_batch_size, TimeInt exec, TimeInt compute) {
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                     triton_model_instance_, total_batch_size, exec.start, compute.start,
                     compute.end, exec.end),
                 "Failed reporting batch statistics.");
  }

  /** Run inference for a given \p request and prepare a response. */
  RequestMeta ProcessRequest(TritonResponseView response, TritonRequestView request) {
    RequestMeta ret;

    auto dali_inputs = GenerateInputs(request);
    ret.batch_size = dali_inputs[0].meta.shape.num_samples();  // Batch size is expected to be the
                                                               // same in every input
    Timer timer{};
    auto outputs_info = dali_executor_->Run(dali_inputs);
    ret.compute_interval = timer.Interval();
    auto dali_outputs = AllocateOutputs(request, response, outputs_info);
    dali_executor_->PutOutputs(dali_outputs);
    return ret;
  }

  /** @brief Generate descriptors of inputs provided by a given request. */
  std::vector<IDescr> GenerateInputs(TritonRequestView request) {
    uint32_t input_cnt = request.InputCount();
    std::vector<IDescr> ret;
    ret.reserve(input_cnt);
    for (uint32_t input_idx = 0; input_idx < input_cnt; ++input_idx) {
      auto input = request.InputByIdx(input_idx);
      auto input_byte_size = input.ByteSize();
      auto input_buffer_count = input.BufferCount();
      std::vector<IBufferDescr> buffers;
      buffers.reserve(input_buffer_count);
      for (uint32_t buffer_idx = 0; buffer_idx < input_buffer_count; ++buffer_idx) {
        auto buffer = input.GetBuffer(buffer_idx, device_type_t::CPU, device_id_);
        buffers.push_back(buffer);
      }
      ret.push_back({input.Meta(), std::move(buffers)});
    }
    return ret;
  }

  /**
   * @brief Allocate outputs required by a given request.
   *
   * Lifetime of the created buffer is bound to the \p response
   */
  std::vector<ODescr> AllocateOutputs(TritonRequestView request, TritonResponseView response,
                                      const std::vector<OutputInfo>& outputs_info) {
    uint32_t output_cnt = request.OutputCount();
    ENFORCE(outputs_info.size() == output_cnt,
            make_string("Number of outputs in the model configuration (", output_cnt,
                        ") does not match to the number of outputs from DALI pipeline (",
                        outputs_info.size(), ")"));
    const auto& output_indices = dali_model_->GetOutputOrder();
    std::vector<ODescr> outputs(output_cnt);
    outputs.reserve(output_cnt);
    for (uint32_t i = 0; i < output_cnt; ++i) {
      auto name = request.OutputName(i);
      int output_idx = output_indices.at(name);
      IOMeta out_meta{};
      out_meta.name = name;
      out_meta.type = outputs_info[output_idx].type;
      out_meta.shape = outputs_info[output_idx].shape;
      auto buffer = response.AllocateOutputBuffer(out_meta, outputs_info[output_idx].device);
      outputs[output_idx] = {out_meta, {buffer}};
    }
    return outputs;
  }

  TritonError ErrorHandler() {
    TritonError error;
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

  std::unique_ptr<DaliExecutor> dali_executor_;
  DaliModel* dali_model_;
};


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

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton TRITONBACKEND API version: ") +
               std::to_string(api_version_major) + "." + std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("'") + name + "' TRITONBACKEND API version: " +
                                      std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
                                      std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                                         .c_str());


  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                                 "triton backend API version does not support this backend");
  }


  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'").c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInitialize: ") + name +
                                      " (version " + std::to_string(version) + ")")
                                         .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  DaliModel* model_state;
  RETURN_IF_ERROR(DaliModel::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));
  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());
  model_state->ReadOutputsOrder();

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  DaliModel* model_state = reinterpret_cast<DaliModel*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInstanceInitialize: ") +
                                      name + " (" + TRITONSERVER_InstanceGroupKindString(kind) +
                                      " device " + std::to_string(device_id) + ")")
                                         .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  DaliModel* model_state = reinterpret_cast<DaliModel*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  DaliModelInstance* instance_state;
  RETURN_IF_ERROR(DaliModelInstance::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  DaliModelInstance* instance_state = reinterpret_cast<DaliModelInstance*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance,
                                                       TRITONBACKEND_Request** reqs,
                                                       const uint32_t request_count) {
  std::vector<TritonRequest> requests;
  for (uint32_t idx = 0; idx < request_count; ++idx) {
    requests.push_back(TritonRequest(reqs[idx]));
  }
  DaliModelInstance* dali_instance;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&dali_instance)));
  std::vector<TRITONBACKEND_Response*> responses(request_count);

  try {
    dali_instance->Execute(std::move(requests));
  } catch (TritonError& err) { return err.release(); }

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::dali
