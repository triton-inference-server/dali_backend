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

#ifndef DALI_BACKEND_DALI_MODEL_H_
#define DALI_BACKEND_DALI_MODEL_H_

#include "src/model_provider/model_provider.h"
#include "src/utils/triton.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"

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
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
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

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_MODEL_H_
