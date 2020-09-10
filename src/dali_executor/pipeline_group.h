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

#ifndef DALI_BACKEND_DALI_EXECUTOR_PIPELINE_GROUP_H_
#define DALI_BACKEND_DALI_EXECUTOR_PIPELINE_GROUP_H_

#include <utility>
#include <vector>
#include "src/dali_executor/dali_pipeline.h"

namespace triton { namespace backend { namespace dali {

/**
 * Abstracts calling processing in multiple DALI pipelines.
 * This class' API shall reflect DaliPipeline API.
 */
class PipelineGroup {
 public:
  using output_shapes_t = std::vector<TensorListShape<>>;

  /**
   * @param pipelines Pipelines, that compose processing call for multiple pipelines.
   *                  Provided DaliPipelines need to originate from the same serialized pipeline.
   */
  explicit PipelineGroup(std::vector<DaliPipeline *> pipelines) : pipelines_(std::move(pipelines)) {}

  ~PipelineGroup() = default;

  DEFAULT_COPY_MOVE_ASSIGN(PipelineGroup);

  /**
   * Run the processing
   */
  void Run();

  /**
   * Wait for the outputs
   */
  void Output();

  /**
   * @return number of outputs. This will be consistent for every DaliPipeline provided
   */
  int GetNumOutputs();

  /**
   * @return shape for every output. This will be consistent for every DaliPipeline provided
   */
  std::vector<TensorListShape<>> GetOutputsShape();

  /**
   * @return data type in every output. This will be consistent for every DaliPipeline provided
   */
  std::vector<dali_data_type_t> GetOutputsTypes();

  /**
   * Gathers outputs from every pipeline provided and puts them into single destination
   * buffer. The order of outputs reflects the order of inputs (thus reflects the order of
   * DaliPipelines too)
   */
  void PutOutput(void *destination, int output_idx, device_type_t destination_device);

  /**
   * Set the input to the DaliPipelines.
   *
   * Input batch will be split across DaliPipelines, according to their capabilities.
   *
   * @param ptr buffer with the input
   * @param name name of the input. This should match the name of the input operator in DALI pipeline
   * @param source_device device of the input data
   * @param data_type type of the input data
   * @param input_shape shape of the input data (input batch)
   */
  void SetInput(const void *ptr, const char *name, device_type_t source_device,
                dali_data_type_t data_type, TensorListShape<> input_shape);

  std::vector<DaliPipeline *> pipelines_;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_PIPELINE_GROUP_H_
