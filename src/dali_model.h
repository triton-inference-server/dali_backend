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

#include "src/dali_executor/dali_pipeline.h"
#include "src/model_provider/model_provider.h"
#include "src/parameters.h"
#include "src/utils/triton.h"
#include "src/utils/utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"

namespace triton { namespace backend { namespace dali {

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

    std::stringstream model_path_ss;
    model_path_ss << model_repo_path << sep << version_ << sep;
    std::string model_path = model_path_ss.str();

    std::stringstream default_model, fallback_model;
    default_model << model_path << GetModelFilename();
    fallback_model << model_path << fallback_model_filename_;

    LoadModel(default_model.str(), fallback_model.str());
  }

  /**
   * Loads a model via the ModelProvider.
   *
   * @param default_model_filename The full path of the model file, specified by the user in the
   *                               model configuration.
   * @param fallback_model_filename The full path of the model file, which is a fallback from the
   *                                default path, in case it is unavailable. The fallback model
   *                                may only represent unserialized model (which will be a subject
   *                                for autoserialization).
   */
  void LoadModel(std::string default_model_filename, std::string fallback_model_filename) {
    std::string& model_filename = default_model_filename;
    std::string target = make_string("/tmp/serialized.model.", timestamp(), ".dali");
    bool load_succeeded = false;

    // Try to load model from the default location
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (make_string("Attempting to load DALI pipeline from file: ", default_model_filename)
                     .c_str()));

    // First try to load the serialized model from the default location
    if (!load_succeeded && TryLoadModel<FileModelProvider>(model_filename))
      load_succeeded = true;

    // Serialized model could not be loaded, try to autoserialize model from the default location.
    if (!load_succeeded && TryLoadModel<AutoserializeModelProvider>(model_filename, target))
      load_succeeded = true;

    if (!load_succeeded) {
      // The default location failed, try to load model from the fallback location.
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (make_string("Attempting to load DALI pipeline from file: ", fallback_model_filename)
               .c_str()));

      model_filename = fallback_model_filename;
      // Fallback location may only represent the unserialized model.
      if (TryLoadModel<AutoserializeModelProvider>(model_filename, target))
        load_succeeded = true;
    }

    if (!load_succeeded) {
      // The fallback location failed, there's nothing we can do more.
      throw std::runtime_error(
          make_string("Failed to load model file. The program looked in the following locations: ",
                      default_model_filename, ", ", fallback_model_filename,
                      ". Please make sure that the model exists in any of the locations and is "
                      "properly serialized or can be properly serialized."));
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (make_string("DALI pipeline from file ", model_filename, " loaded successfully.").c_str()));
  }


  /**
   * Try to load a model with given ModelProvider and args to its constructor.
   *
   * @tparam ModelProvider ModelProvider type used to load given model.
   * @tparam Args Arguments to the ModelProvider constructor.
   * @param args Arguments to the ModelProvider constructor.
   * @return True, if the model has been loaded successfully.
   */
  template<typename ModelProvider, typename... Args>
  bool TryLoadModel(const Args&... args) {
    try {
      auto mp = std::make_unique<ModelProvider>(args...);
      if (DaliPipeline::ValidateDaliPipeline(mp->GetModel())) {
        dali_model_provider_ = std::move(mp);
        return true;
      }
    } catch (const std::runtime_error& e) {
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                  (make_string("Loading model failed: ", e.what()).c_str()));
    }
    return false;
  }


  std::string GetModelFilename() {
    std::string ret;
    TRITON_CALL_GUARD(model_config_.MemberAsString("default_model_filename", &ret));
    return ret.empty() ? "model.dali" : ret;
  }


  ModelParameters params_;
  std::unique_ptr<ModelProvider> dali_model_provider_;
  std::unordered_map<std::string, int> output_order_;
  const std::string fallback_model_filename_ = "dali.py";
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_MODEL_H_
