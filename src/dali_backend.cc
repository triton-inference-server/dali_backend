// The MIT License (MIT)
//
// Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES
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

#include "src/dali_model.h"
#include "src/dali_model_instance.h"
#include "src/utils/triton.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace dali {

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  TimeRange tr("[DALI BE] Initialize", TimeRange::kNavy);
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


  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &backend_config_message));
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());
  BackendParameters backend_params(make_string(buffer));

  try {
    DaliPipeline::LoadPluginLibs(backend_params.GetPluginNames());
  } catch (const DaliBackendException& e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                make_string("Failed to load plugin libs: ", e.what()).c_str());
  }

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
  TimeRange tr("[DALI BE] Finalize", TimeRange::kNavy);
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
  TimeRange tr("[DALI BE] ModelInitialize", TimeRange::kNavy);
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

  if (model_state->ShouldAutoCompleteConfig()) {
    RETURN_IF_ERROR(model_state->AutoCompleteConfig());
  }

  RETURN_IF_ERROR(model_state->ValidateModelConfig());
  model_state->ReadOutputsOrder();

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  TimeRange tr("[DALI BE] ModelFinalize", TimeRange::kNavy);
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
  TimeRange tr("[DALI BE] ModelInstanceInitialize", TimeRange::kNavy);
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
  TimeRange tr("[DALI BE] ModelInstanceFinalize", TimeRange::kNavy);
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
  TimeRange tr("[DALI BE] ModelInstanceExecute", TimeRange::kNavy);
  std::vector<TritonRequest> requests;
  for (uint32_t idx = 0; idx < request_count; ++idx) {
    requests.emplace_back(reqs[idx]);
  }
  DaliModelInstance* dali_instance;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&dali_instance)));
  std::vector<TRITONBACKEND_Response*> responses(request_count);

  try {
    dali_instance->Execute(requests);
  } catch (TritonError& err) { return err.release(); }

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::dali
