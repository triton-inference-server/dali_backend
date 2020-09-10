// The MIT License (MIT)
//
// Copyright (c) 2020 NVIDIA CORPORATION
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <memory>
#include "src/dali_backend.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace dali {

 class DaliModel : public ::triton::backend::BackendModel {
 public:
  static TRITONSERVER_Error *Create(
          TRITONBACKEND_Model *triton_model, DaliModel **state);

  virtual ~DaliModel() = default;

  TRITONSERVER_Error* ValidateModelConfig()
  {
    // We have the json DOM for the model configuration...
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
    LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("model configuration:\n") + buffer.Contents()).c_str());

    return nullptr;  // success
  }

  const ModelProvider &GetModelProvider() const { return *dali_model_provider_; };

 private:
  explicit DaliModel(TRITONBACKEND_Model *triton_model) : BackendModel(triton_model) {
    const char sep = '/';

    const char *model_repo_path;
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
    model_config_.MemberAsString("default_model_filename", &ret);
    return ret.empty() ? "model.dali" : ret;
  }


  std::unique_ptr<ModelProvider> dali_model_provider_;
};


TRITONSERVER_Error *
DaliModel::Create(TRITONBACKEND_Model *triton_model, DaliModel **state) {
  try {
    *state = new DaliModel(triton_model);
  }
  catch (const BackendModelException &ex) {
    RETURN_ERROR_IF_TRUE(
            ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
            std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }


  return nullptr;  // success
}


class DaliModelInstance : public ::triton::backend::BackendModelInstance {
 public:
  static TRITONSERVER_Error *
  Create(DaliModel *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
         DaliModelInstance **state);

  DaliExecutor &GetDaliExecutor() { return *dali_executor_; }

 private:
  DaliModelInstance(DaliModel *model, TRITONBACKEND_ModelInstance *triton_model_instance)
          : BackendModelInstance(model, triton_model_instance) {
    dali_executor_ = std::make_unique<DaliExecutor>(model->GetModelProvider().GetModel(),
                                                    device_id_);
  }


  std::unique_ptr<DaliExecutor> dali_executor_;
};


TRITONSERVER_Error *
DaliModelInstance::Create(
        DaliModel *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
        DaliModelInstance **state) {
  const char *instance_name;
  RETURN_IF_ERROR(
          TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
          TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
          TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new DaliModelInstance(model_state, triton_model_instance);
  return nullptr;  // success
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error *
TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend) {
  const char *cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
          TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Triton TRITONBACKEND API version: ") +
           std::to_string(api_version_major) + "." +
           std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());


  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }


  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  DaliModel* model_state;
  RETURN_IF_ERROR(DaliModel::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  DaliModel* model_state = reinterpret_cast<DaliModel*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
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
  RETURN_IF_ERROR(
          DaliModelInstance::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  DaliModelInstance* instance_state =
      reinterpret_cast<DaliModelInstance*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error *TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                                       TRITONBACKEND_Request **reqs,
                                                       const uint32_t request_count) {
  TRITONSERVER_Error *error = nullptr;  // success
  DaliModelInstance *instance_state;
  RETURN_IF_ERROR(
          TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&instance_state)));
  std::vector<TRITONBACKEND_Request *> requests(reqs, reqs + request_count);
  std::vector<TRITONBACKEND_Response *> responses(request_count);

  for (size_t i = 0; i < responses.size(); i++) {
    //TODO Do not process requests one by one, but gather all
    //     into one buffer and process it in DALI all together
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&responses[i], requests[i]));

    try {
      detail::ProcessRequest(responses[i], requests[i], instance_state->GetDaliExecutor());
    } catch (DaliBackendException &e) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
      error = TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNKNOWN,
                                    make_string("DALI Backend error: ", e.what()).c_str());
    } catch (DALIException &e) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
      error = TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNKNOWN,
                                    make_string("DALI error: ", e.what()).c_str());
    } catch (std::runtime_error &e) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (e.what()));
      error = TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNKNOWN,
                                    make_string("runtime error: ", e.what()).c_str());
    } catch (...) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("Unknown error"));
      error = TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_UNKNOWN,
                                    "Unknown DALI Backend error");
    }

    RETURN_IF_ERROR(
            TRITONBACKEND_ResponseSend(responses[i],
                                       TRITONSERVER_ResponseCompleteFlag::TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                                       error));
    RETURN_IF_ERROR(
            TRITONBACKEND_RequestRelease(requests[i],
                                         TRITONSERVER_RequestReleaseFlag::TRITONSERVER_REQUEST_RELEASE_ALL));
  }
  return nullptr;
}

}  // extern "C"

}}} // namespace triton::backend::dali
