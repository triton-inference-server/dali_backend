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

#ifndef DALI_BACKEND_DALI_EXECUTOR_DALI_EXECUTOR_H_
#define DALI_BACKEND_DALI_EXECUTOR_DALI_EXECUTOR_H_

#include <map>
#include <string>
#include <utility>

#include "src/dali_executor/dali_pipeline.h"
#include "src/dali_executor/io_buffer.h"
#include "src/dali_executor/io_descriptor.h"


namespace triton { namespace backend { namespace dali {


struct OutputInfo {
  TensorListShape<> shape;
  dali_data_type_t type;
  device_type_t device;
};

class DaliExecutor {
 public:
  DaliExecutor(DaliPipeline pipeline) :
      pipeline_(std::move(pipeline)), thread_pool_(GetNumThreads(), pipeline_.DeviceId(), false) {}

  /**
   * Run DALI pipeline and return the result descriptor
   */
  std::vector<OutputInfo> Run(const std::vector<IDescr>& inputs);


  void PutOutputs(const std::vector<ODescr>& outputs);

 private:
  void SetupInputs(const std::vector<IDescr>& inputs);

  /**
   * Schedule copy to a continous buffer and return IDecr to the new buffer.
   */
  IDescr ScheduleInputCopy(const IDescr& buffers);

  int GetNumThreads() {
    auto n_threads = pipeline_.NumThreadsArg();
    return (n_threads < 1) ? 1 : n_threads;
  }

  DaliPipeline pipeline_;
  ThreadPool thread_pool_;
  std::map<std::string, IOBuffer<CPU>> cpu_buffers_;
  std::map<std::string, IOBuffer<GPU>> gpu_buffers_;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_DALI_EXECUTOR_H_
