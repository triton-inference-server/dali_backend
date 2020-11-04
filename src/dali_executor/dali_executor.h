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

#include <string>
#include <utility>
#include <vector>
#include "src/dali_executor/pipeline_group.h"
#include "src/dali_executor/pipeline_pool.h"
#include "src/dali_executor/utils/dali.h"


namespace triton { namespace backend { namespace dali {

std::vector<int> distribute_batch_size(int batch_size);


template <typename T>
struct IODescriptor {
  std::string name;
  dali_data_type_t type;
  device_type_t device;
  TensorListShape<> shape;
  span<T> buffer;
};

using InputDescriptor = IODescriptor<const char>;
using OutputDescriptor = IODescriptor<char>;

struct shape_and_type_t {
  TensorListShape<> shape;
  dali_data_type_t type;
};

class DaliExecutor {
 public:
  DaliExecutor(std::string serialized_pipeline, int device_id)
      : serialized_pipeline_(std::move(serialized_pipeline)),
        device_id_(device_id)
  {
  }

  std::vector<shape_and_type_t> Run(const std::vector<InputDescriptor>& inputs);

  void PutOutputs(const std::vector<OutputDescriptor>& outputs);

  size_t NumCreatedPipelines() { return pipeline_pool_.NumCreatedPipelines(); }

 private:
  PipelineGroup SetupInputs(const std::vector<InputDescriptor>& inputs);

  std::string serialized_pipeline_;
  int device_id_;
  PipelinePool pipeline_pool_;
  std::vector<int> batch_sizes_;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_DALI_EXECUTOR_H_
