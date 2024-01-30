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

#ifndef TRITONDALIBACKEND_PARAMETERS_H
#define TRITONDALIBACKEND_PARAMETERS_H

#include "src/utils/triton.h"
#include "src/utils/utils.h"


namespace triton { namespace backend { namespace dali {

#if defined(_WIN32)
const std::string separator = ";";
#else
const std::string separator = ":";
#endif

class ModelParameters {
 public:
  explicit ModelParameters(common::TritonJson::Value& model_config) {
    model_config.MemberAsObject("parameters", &params_);
  }

  /**
   * Return a value of a parameter with a given `key`
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

  std::vector<std::string> GetOutputsToSplit() {
    std::string outs_list = GetParam<std::string>("split_along_outer_axis");
    return split(outs_list, separator);
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

  common::TritonJson::Value params_{};
};


class BackendParameters {
 public:
   explicit BackendParameters(const std::string& backend_config_json) {
     FromConfigJson(backend_config_json);
   }

   explicit BackendParameters(TRITONBACKEND_Backend* backend) {
     FromBackend(backend);
   }

   explicit BackendParameters(TRITONBACKEND_Model* model) {
     TRITONBACKEND_Backend* backend;
     TRITON_CALL_GUARD(TRITONBACKEND_ModelBackend(model, &backend));
     FromBackend(backend);
   }

  /**
   * Return a value of a parameter with a given `key`
   * or `def` if the parameter is not present.
   */
  template<typename T>
  T GetParam(const std::string& key, const T& def = T()) const {
    T result = def;
    GetMember(key, result);
    return result;
  }

  std::vector<std::string> GetPluginNames() const {
    auto plugin_list = GetParam<std::string>("plugin_libs");
    return split(plugin_list, separator);
  }

  bool ShouldReleaseBuffersAfterUnload() const {
    return GetParam<bool>("release_after_unload");
  }

 private:
  template<typename T>
  void GetMember(const std::string& key, T& value) const {
    auto key_c = key.c_str();
    if (params_.Find(key_c)) {
      std::string string_value{};
      TRITON_CALL_GUARD(params_.MemberAsString(key_c, &string_value));
      value = from_string<T>(string_value);
    }
  }

  void FromBackend(TRITONBACKEND_Backend* backend) {
    TRITONSERVER_Message* backend_config_message;
    TRITON_CALL_GUARD(TRITONBACKEND_BackendConfig(backend, &backend_config_message));
    const char* buffer;
    size_t byte_size;
    TRITON_CALL_GUARD(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());
    auto backend_config_json = make_string(buffer);
    FromConfigJson(backend_config_json);
  }

  void FromConfigJson(const std::string& backend_config_json) {
    TRITON_CALL_GUARD(config_.Parse(backend_config_json));
    if (config_.Find("cmdline")) {
      TRITON_CALL_GUARD(config_.MemberAsObject("cmdline", &params_));
    }
  }

  // WAR: Keeping the config_ here shouldn't be necessary. However, due to
  //      an issue in TritonJson implementation, when top-level TritonJson::Value
  //      goes out of scope, in the remaining child Values, first 8 bytes are trimmed.
  common::TritonJson::Value config_{};
  common::TritonJson::Value params_{};
};


}}}  // namespace triton::backend::dali

#endif  // TRITONDALIBACKEND_PARAMETERS_H
