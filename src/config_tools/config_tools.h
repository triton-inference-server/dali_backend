// The MIT License (MIT)
//
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES
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

#ifndef DALI_BACKEND_CONFIG_TOOLS_CONFIG_TOOLS_H_
#define DALI_BACKEND_CONFIG_TOOLS_CONFIG_TOOLS_H_

#include <optional>

#include "src/utils/triton.h"
#include "src/utils/utils.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace dali {

using triton::common::TritonJson;

struct IOConfig {
  std::string name;
  dali_data_type_t dtype;
  std::optional<std::vector<int64_t>> shape;

  IOConfig() = default;

  explicit IOConfig(const std::string &name,
                    dali_data_type_t dtype = DALI_NO_TYPE,
                    std::optional<std::vector<int64_t>> shape = {})
    : name(name)
    , dtype(dtype)
    , shape(shape) {}
};

/**
 * @brief Convert DALI data type to type string used in Triton model config
 */
std::string to_triton_config(dali_data_type_t type);


/**
 * @brief Fill TritonJson array with values from dims.
 * Initial size of array must be smaller or equal dims.size()
 */
void SetShapeArray(TritonJson::Value &array, const std::vector<int64_t> &dims);


/**
 * @brief Find object in array that has a name field equal to given string.
 *  If found, the object is assigned to *ret and the index in array is returned.
 *  The previous value of *ret is released.
 *  If not found, *ret is not modified and an empty optional is returned.
 */
std::optional<size_t> FindObjectByName(TritonJson::Value &array, const std::string &name,
                                       TritonJson::Value *ret);


/**
 * @brief Read an array of ints and return it as a vector.
 */
std::vector<int64_t> ReadShape(TritonJson::Value &dims_array);


/**
 * @brief Match shapes from config file and pipeline and return the result of matching.
 *
 * e.g. shapes [-1, 2, -1] and [-1, -1, 3] will match to [-1, 2, 3]
 *
 * Throws an error when shapes cannot be matched.
 */
std::vector<int64_t> MatchShapes(const std::string &name,
                                 const std::vector<int64_t> &config_shape,
                                 const std::vector<int64_t> &pipeline_shape);


/**
 * @brief Validates data_type field in IO object against provided value.
 * Returns data type to be used for autofill.
 */
std::string AutofillDtypeConfig(TritonJson::Value &io_object, const std::string &name,
                                dali_data_type_t dtype);


/**
 * @brief Validates data_type field in IO object against provided value.
 */
void ValidateDtypeConfig(TritonJson::Value &io_object, const std::string &name,
                         dali_data_type_t dtype);


/**
 * @brief Validates dims field in IO object again provided value.
 * Matches the shape from IO object with the provided shape
 * and outputs the result of matching to resulting_dims.
 */
void AutofillShapeConfig(TritonJson::Value &io_object, const std::string &name,
                         const std::optional<std::vector<int64_t>> &shape,
                         TritonJson::Value &resulting_dims);

/**
 * @brief Validates dims field in IO object again provided value.
 */
void ValidateShapeConfig(TritonJson::Value &io_object, const std::string &name,
                         const std::optional<std::vector<int64_t>> &shape);

/**
 * @brief Validates IO object against provided config values
 * and outputs auto-configured IO object to new_io_object.
 */
void AutofillIOConfig(TritonJson::Value &io_object, const IOConfig &io_config,
                      TritonJson::Value &new_io_object);


/**
 * @brief Validates IO object against provided config values.
 */
void ValidateIOConfig(TritonJson::Value &io_object, const IOConfig &io_config);


/**
 * @brief Validates inputs array against provided config values.
 * Outputs auto-configured array to new_inputs.
 *
 * Auto-filled config keeps the order of inputs given in the original config file.
 * If an input was not named in the original config, it is appended at the end.
 * Inputs that do not appear in the original config are ordered lexicographically (pipeline order).
 */
void AutofillInputsConfig(TritonJson::Value &inputs, const std::vector<IOConfig> &in_configs,
                          TritonJson::Value &new_inputs);


/**
 * @brief Validates outputs array against provided config values
 * Outputs auto-configured array to new_outputs.
 *
 * It's assumed that outputs in the config file have the same order as those in the DALI pipeline.
 * If the config sets the name of an output, it overrides the name specified in the DALI pipeline.
 */
void AutofillOutputsConfig(TritonJson::Value &outputs, const std::vector<IOConfig> &out_configs,
                           TritonJson::Value &new_outputs);


void AutofillConfig(TritonJson::Value &config, const std::vector<IOConfig> &in_configs,
                    const std::vector<IOConfig> &out_configs, int pipeline_max_batch_size);


/**
 * @brief Validate outputs array against provided config values
 *
 * Names of the outputs in the config file do not need to match the names of the outputs
 * specified in the DALI pipeline.
 */
void ValidateOutputs(TritonJson::Value &outs, const std::vector<IOConfig> &out_configs);


/**
 * @brief Validate inputs array against provided config values.
 *
 * Names of the inputs in the config file must match the names of the inputs in the pipeline.
 */
void ValidateInputs(TritonJson::Value &ins, const std::vector<IOConfig> &in_configs);


/**
 * @brief Read max_batch_size field from the config. Return -1 if the field is missing.
 */
int ReadMaxBatchSize(TritonJson::Value &config);


/**
 * @brief Validate the model max batch size and inputs and outputs configs against provided values.
 */
void ValidateConfig(TritonJson::Value &config, const std::vector<IOConfig> &in_configs,
                    const std::vector<IOConfig> &out_configs);

}}} // namespace triton::backend::dali

#endif  // DALI_BACKEND_CONFIG_TOOLS_CONFIG_TOOLS_H_
