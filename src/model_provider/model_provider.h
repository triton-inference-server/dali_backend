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

#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "src/utils/cmake_to_cpp.h"

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


namespace detail {


inline void ValidatePath(const std::string& path) {
  // Whitelist of allowed characters for a file path.
  const std::string allowed_chars =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-";
  for (char c : path) {
    if (allowed_chars.find(c) == std::string::npos) {
      throw std::runtime_error(
          "Invalid character found in model path. The path contains a forbidden character: '" +
          std::string(1, c) + "'");
    }
  }

  // Check for path traversal sequences to prevent reading/writing outside intended directories.
  if (path.find("..") != std::string::npos) {
    throw std::runtime_error("Invalid sequence '..' found in path (Path Traversal attempt).");
  }
}

inline std::string GenerateAutoserializeScript(const std::string& module_path,
                                               const std::string& target_file_path) {
  std::stringstream py_script;
  py_script << "import importlib, sys\n"
            << "from nvidia.dali._utils.autoserialize import invoke_autoserialize\n"
            << "spec = importlib.util.spec_from_file_location('autoserialize_mod', r'\""
            << module_path << "\"')\n"
            << "head_module = importlib.util.module_from_spec(spec)\n"
            << "sys.modules['autoserialize_mod'] = head_module\n"
            << "spec.loader.exec_module(head_module)\n"
            << "invoke_autoserialize(head_module, r'\"" << target_file_path << "\"')";
  return py_script.str();
}


inline void CallCmdSecure(const std::string& command, const std::vector<std::string>& args) {
  // Convert vector of string arguments to a C-style array of char pointers for execvp.
  std::vector<char*> argv;
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);  // The array must be null-terminated.

  pid_t pid = fork();
  if (pid == -1) {
    // Fork failed
    throw std::runtime_error("Failed to fork process.");
  } else if (pid == 0) {
    // Child process
    execvp(command.c_str(), argv.data());
    // If execvp returns, it means an error occurred.
    perror("execvp");  // Print error message to stderr
    exit(EXIT_FAILURE);
  }

  // Parent process
  int status;
  if (waitpid(pid, &status, 0) == -1) {
    throw std::runtime_error("Failed to wait for child process.");
  }

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    // Command succeeded
    return;
  }

  // Command failed
  std::stringstream ss;
  ss << "Command failed. ";
  if (WIFEXITED(status)) {
    ss << "Exited with error code: " << WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    ss << "Killed by signal: " << WTERMSIG(status);
  }
  throw std::runtime_error(ss.str());
}

}  // namespace detail


class AutoserializeModelProvider : public ModelProvider {
 public:
  AutoserializeModelProvider() = default;

  AutoserializeModelProvider(const std::string& module_path, const std::string& target_file) {
    detail::ValidatePath(module_path);
    detail::ValidatePath(target_file);
    std::string python_script = detail::GenerateAutoserializeScript(module_path, target_file);

    std::vector<std::string> args = {
        "python3",     // Arg 0: the program name itself
        "-c",          // Arg 1: flag to execute a script string
        python_script  // Arg 2: the script to execute
    };

    // Call the secure command execution function
    detail::CallCmdSecure("python3", args);
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
