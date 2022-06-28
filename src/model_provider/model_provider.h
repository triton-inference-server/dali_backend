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

#ifndef DALI_BACKEND_MODEL_PROVIDER_MODEL_PROVIDER_H_
#define DALI_BACKEND_MODEL_PROVIDER_MODEL_PROVIDER_H_

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace triton { namespace backend { namespace dali {

class ModelProvider {
 public:
  virtual const std::string& GetModel() const = 0;

  virtual ~ModelProvider() = default;
};


template<typename ModelFunctor>
class FunctorModelProvider : public ModelProvider {
 public:
  template<typename... Args>
  explicit FunctorModelProvider(Args&&... args) :
      functor_(), serialized_model_(functor_(std::forward<Args>(args)...)) {}

  const std::string& GetModel() const override {
    return serialized_model_;
  }

  ~FunctorModelProvider() override = default;

  ModelFunctor functor_;
  std::string serialized_model_;
};


class FileModelProvider : public ModelProvider {
 public:
  FileModelProvider() = default;

  explicit FileModelProvider(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
      throw std::runtime_error(std::string("Failed to open model file: ") + filename);
    std::stringstream ss;
    ss << fin.rdbuf();
    model_ = ss.str();
  }

  const std::string& GetModel() const override {
    return model_;
  }

  ~FileModelProvider() override = default;

 private:
  std::string model_ = {};
};

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace detail {

inline std::string GenerateAutoserializeCmd(const std::string& module_path,
                                            const std::string& target_file_path) {
  std::stringstream cmd;
  cmd << STRINGIFY(CONDA_ENVIRONMENT) << "/bin/";
  cmd << R"py(python3 -c "
import importlib, sys
from nvidia.dali._utils.autoserialize import invoke_autoserialize
spec = importlib.util.spec_from_file_location('autoserialize_mod', ')py"
      << module_path << R"py(')
head_module = importlib.util.module_from_spec(spec)
sys.modules['autoserialize_mod'] = head_module
spec.loader.exec_module(head_module)
invoke_autoserialize(head_module, ')py"
      << target_file_path << R"py(')
")py";

  return cmd.str();
}

inline void CallSystemCmd(const std::string& cmd) {
  auto status = system(cmd.c_str());

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    // Succeed
    return;

  // Failed
  std::stringstream ss;
  ss << "Failed to call system command. ";
  if (WIFEXITED(status)) {
    ss << "Exited with error code: " << WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    ss << "Killed by signal: " << WTERMSIG(status);
  } else if (WIFSTOPPED(status)) {
    ss << "Stopped by signal: " << WSTOPSIG(status);
  } else if (WIFCONTINUED(status)) {
    ss << "Continued";
  }
  throw std::runtime_error(ss.str());
}

}  // namespace detail


class AutoserializeModelProvider : public ModelProvider {
 public:
  AutoserializeModelProvider() = default;

  AutoserializeModelProvider(const std::string& module_path, const std::string& target_file) {
    auto cmd = detail::GenerateAutoserializeCmd(module_path, target_file);
    detail::CallSystemCmd(cmd);
    fmp_ = FileModelProvider(target_file);
  }

  const std::string& GetModel() const override {
    return fmp_.GetModel();
  }

  ~AutoserializeModelProvider() override = default;

 private:
  FileModelProvider fmp_{};
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_MODEL_PROVIDER_MODEL_PROVIDER_H_
