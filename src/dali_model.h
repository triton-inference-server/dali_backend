// The MIT License (MIT)
//
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES
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

#include <atomic>

#include "parameters.h"
#include "src/config_tools/config_tools.h"
#include "src/dali_executor/dali_pipeline.h"
#include "src/dali_executor/utils/dali.h"
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

  DaliModel(DaliModel&&) = delete;

  virtual ~DaliModel() {
    bool last_loaded_model = --loaded_models_count_ == 0;
    if (last_loaded_model && ShouldReleaseBuffersAfterUnload()) {
      ReleaseUnusedMemory();
    }
    assert(loaded_models_count_ >= 0);
  };

  TRITONSERVER_Error* ValidateModelConfig() {
    // We have the json DOM for the model configuration...
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("model configuration:\n") + buffer.Contents()).c_str());
    try {
      ValidateConfig(model_config_, pipeline_inputs_, pipeline_outputs_, Batched());
    } catch (TritonError& err) {
      return err.release();
    }

    return nullptr;  // success
  }

  const ModelProvider& GetModelProvider() const {
    return *dali_model_provider_;
  };

  ModelParameters& GetModelParamters() {
    return params_;
  }

  bool IsOutputSplit(const std::string& name) {
    auto it = std::find(outputs_to_split_.begin(), outputs_to_split_.end(), name);
    return it != outputs_to_split_.end();
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

  bool ShouldAutoCompleteConfig() const {
    return should_auto_complete_config_;
  }

  bool ShouldReleaseBuffersAfterUnload() const {
    return backend_params_.ShouldReleaseBuffersAfterUnload();
  }

  TRITONSERVER_Error* AutoCompleteConfig() {
    try {
      AutofillConfig(model_config_, pipeline_inputs_, pipeline_outputs_, pipeline_max_batch_size_,
                     Batched());
      TRITON_CALL(SetModelConfig());
    } catch (TritonError& err) {
      return err.release();
    }
    return nullptr;
  }

  bool Batched() const {
    return !(config_max_batch_size_.has_value() && config_max_batch_size_ == 0);
  }

 private:
  explicit DaliModel(TRITONBACKEND_Model* triton_model) :
      BackendModel(triton_model), params_(model_config_), backend_params_(triton_model) {
    const char sep = '/';

    const char* model_repo_path;
    TRITONBACKEND_ArtifactType artifact_type;
    TRITON_CALL(TRITONBACKEND_ModelRepository(triton_model_, &artifact_type, &model_repo_path));

    std::string config_path = make_string(model_repo_path, sep, "config.pbtxt");
    std::ifstream config_file(config_path);
    if (config_file.good()) {
      std::stringstream config_buffer;
      config_buffer << config_file.rdbuf();
      auto config_text = config_buffer.str();
      config_max_batch_size_ = ReadMBSFromPBtxt(config_text);
    }

    std::string model_path = make_string(model_repo_path, sep, version_, sep);

    std::stringstream default_model, fallback_model;
    default_model << model_path << GetModelFilename();
    fallback_model << model_path << fallback_model_filename_;

    ReadParams();

    LoadModel(default_model.str(), fallback_model.str());
    ReadPipelineProperties();

    TRITON_CALL(
        TRITONBACKEND_ModelAutoCompleteConfig(triton_model_, &should_auto_complete_config_));
    ++loaded_models_count_;
  }

  void ReadParams() {
    outputs_to_split_ = params_.GetOutputsToSplit();
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
    std::string target = tmp_model_file();
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

  /**
   * @brief Returns a device id on which DALI pipeline can be instantiated,
   *   or CPU_ONLY_DEVICE_ID if only cpu is available
   */
  int FindDevice() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          make_string("DALI autoconfig -- found ", dev_count, " available devices").c_str());
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, make_string("DALI autoconfig -- no available devices. "
                                                        "Using CPU_ONLY_DEVICE")
                                                .c_str());
      return CPU_ONLY_DEVICE_ID;
    }
    triton::common::TritonJson::Value instance_groups;
    bool found_inst_groups = model_config_.Find("instance_group", &instance_groups);
    int config_dev = -1;
    if (found_inst_groups) {
      config_dev = ReadDeviceFromInstanceGroups(instance_groups);
    }
    if (config_dev != -1) {
      return config_dev;
    }
    // config doesn't specify any GPU so we can choose any available
    return 0;
  }

  // This method tries to find any instance group and select any device id from it.
  // If there is any cpu inst. group, CPU_ONLY_DEVICE_ID is selected. If not,
  // any gpu from any gpu inst. group is selected.
  // If there are no instance groups defined, -1 is returned.
  int ReadDeviceFromInstanceGroups(triton::common::TritonJson::Value& inst_groups) {
    TRITON_CALL(inst_groups.AssertType(triton::common::TritonJson::ValueType::ARRAY));
    auto count = inst_groups.ArraySize();
    for (size_t i = 0; i < count; ++i) {
      triton::common::TritonJson::Value inst_group;
      inst_groups.IndexAsObject(i, &inst_group);
      int64_t instance_count = 0;
      inst_group.MemberAsInt("count", &instance_count);
      if (instance_count == 0) {
        continue;  // instance group is empty
      }
      std::string kind_str;
      if (inst_group.MemberAsString("kind", &kind_str) == TRITONJSON_STATUSSUCCESS) {
        if (kind_str == "KIND_GPU") {
          triton::common::TritonJson::Value dev_array;
          if (inst_group.Find("gpus", &dev_array) && dev_array.ArraySize() > 0) {
            int64_t found_gpu = -1;
            dev_array.IndexAsInt(0, &found_gpu);
            if (found_gpu >= 0 && found_gpu <= std::numeric_limits<int>::max()) {
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  make_string("DALI autoconfig -- found device defined in the config file: ",
                              found_gpu)
                      .c_str());
              return static_cast<int>(found_gpu);
            }
          }
        }
      }
    }
    return -1;
  }

  void ReadPipelineProperties() {
    auto mbs_arg = config_max_batch_size_.value_or(-1);
    if (mbs_arg == 0) {
      mbs_arg = -1;
    }
    auto pipeline = InstantiateDaliPipeline(mbs_arg);
    auto input_names = pipeline.ListInputs();
    pipeline_inputs_.resize(input_names.size());
    for (size_t i = 0; i < input_names.size(); ++i) {
      auto name = input_names[i];
      pipeline_inputs_[i].name = name;
      pipeline_inputs_[i].dtype = pipeline.GetInputType(name);
      pipeline_inputs_[i].shape = pipeline.GetInputShape(name);
    }

    int num_outputs = pipeline.GetNumOutput();
    pipeline_outputs_.resize(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      pipeline_outputs_[i].name = pipeline.GetOutputName(i);
      pipeline_outputs_[i].dtype = pipeline.GetDeclaredOutputType(i);
      auto shape = pipeline.GetDeclaredOutputShape(i);
      if (shape && IsOutputSplit(pipeline_outputs_[i].name)) {
        shape->erase(shape->begin());  // remove outer dimension
      }
      pipeline_outputs_[i].shape = shape;
    }
    pipeline_max_batch_size_ = pipeline.GetMaxBatchSize();
  }

  DaliPipeline InstantiateDaliPipeline(int config_max_batch_size) {
    int device_id = FindDevice();
    const std::string& serialized_pipeline = GetModelProvider().GetModel();
    try {
      return DaliPipeline(serialized_pipeline, config_max_batch_size, 1, device_id);
    } catch (const DALIException& e) {
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                  make_string("DALI autoconfig -- failed to instantiate pipeline on device ",
                  device_id, ": ", e.what()).c_str());
      return DaliPipeline(serialized_pipeline, config_max_batch_size, 1, CPU_ONLY_DEVICE_ID);
    }
  }

  ModelParameters params_;
  BackendParameters backend_params_;
  std::vector<std::string> outputs_to_split_;
  std::unique_ptr<ModelProvider> dali_model_provider_;
  std::unordered_map<std::string, int> output_order_;
  std::vector<IOConfig> pipeline_inputs_{};
  std::vector<IOConfig> pipeline_outputs_{};
  int pipeline_max_batch_size_ = -1;
  std::optional<int> config_max_batch_size_ = {};
  const std::string fallback_model_filename_ = "dali.py";
  bool should_auto_complete_config_ = false;
  static std::atomic_int loaded_models_count_;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_MODEL_H_
