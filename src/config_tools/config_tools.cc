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

#include "src/config_tools/config_tools.h"

#include <limits>

namespace triton { namespace backend { namespace dali {


/**
 * @brief convert DALI data type to type string used in Triton model config
 */
std::string to_triton_config(dali_data_type_t type) {
  switch (type) {
    case DALI_UINT8   :
      return "TYPE_UINT8";
    case DALI_UINT16  :
      return "TYPE_UINT16";
    case DALI_UINT32  :
      return "TYPE_UINT32";
    case DALI_UINT64  :
      return "TYPE_UINT64";
    case DALI_INT8    :
      return "TYPE_INT8";
    case DALI_INT16   :
      return "TYPE_INT16";
    case DALI_INT32   :
      return "TYPE_INT32";
    case DALI_INT64   :
      return "TYPE_INT64";
    case DALI_FLOAT16 :
      return "TYPE_FP16";
    case DALI_FLOAT   :
      return "TYPE_FP32";
    case DALI_FLOAT64 :
      return "TYPE_FP64";
    case DALI_BOOL    :
      return "TYPE_BOOL";
    default:
      return "TYPE_INVALID";
  }
}


void SetShapeArray(TritonJson::Value &array, const std::vector<int64_t> &dims) {
  TRITON_CALL(array.AssertType(TritonJson::ValueType::ARRAY));
  ENFORCE(array.ArraySize() <= dims.size(), "SetShapeArray expects the initial array size to be "
                                            "smaller or equal the number of dimensions.");
  size_t i = 0;
  const auto arr_size = array.ArraySize();
  for (; i < arr_size; ++i) {
    TritonJson::Value elem;
    array.At(i, &elem);
    elem.SetInt(dims[i]);
  }
  for (; i < dims.size(); ++i) {
    array.AppendInt(dims[i]);
  }
}


std::optional<size_t> FindObjectByName(TritonJson::Value &array, const std::string &name,
                                       TritonJson::Value *ret) {
  TritonJson::Value obj;
  size_t len = array.ArraySize();
  for (size_t i = 0; i < len; ++i) {
    array.IndexAsObject(i, &obj);
    std::string found_name;
    TritonError err{obj.MemberAsString("name", &found_name)};
    if (!err && found_name == name) {
      ret->Release();
      array.IndexAsObject(i, ret);
      return {i};
    }
  }
  return std::nullopt;
}


std::vector<int64_t> ReadShape(TritonJson::Value &dims_array) {
  TRITON_CALL(dims_array.AssertType(TritonJson::ValueType::ARRAY));
  size_t len = dims_array.ArraySize();
  std::vector<int64_t> result(len);
  for (size_t i = 0; i < len; ++i) {
    TRITON_CALL(dims_array.IndexAsInt(i, &result[i]));
  }
  return result;
}


std::vector<int64_t> MatchShapes(const std::string &name,
                                 const std::vector<int64_t> &config_shape,
                                 const std::vector<int64_t> &pipeline_shape) {
  if (config_shape.size() != pipeline_shape.size()) {
    throw TritonError::InvalidArg(make_string("Mismatch in number of dimensions for \"", name, "\"\n"
                                  "Number of dimensions defined in config: ", config_shape.size(),
                                  "\nNumber of dimensions defined in pipeline: ",
                                  pipeline_shape.size()));
  }
  std::vector<int64_t> result(config_shape.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (config_shape[i] != pipeline_shape[i]) {
      if (config_shape[i] == -1 || pipeline_shape[i] == -1) {
        result[i] = std::max(config_shape[i], pipeline_shape[i]);
      } else {
        throw TritonError::InvalidArg(
          make_string("Mismath in dims for ", name, "\nDims defined in config: ",
                      vec_to_string(config_shape), "\nDims defined in pipeline: ",
                      vec_to_string(pipeline_shape)));
      }
    } else {
      result[i] = config_shape[i];
    }
  }
  return result;
}


template <bool allow_missing>
std::string ProcessDtypeConfig(TritonJson::Value &io_object, const std::string &name,
                              dali_data_type_t dtype) {
  TritonJson::Value dtype_obj(TritonJson::ValueType::OBJECT);
  if (io_object.Find("data_type", &dtype_obj)) {
    std::string found_dtype;
    TRITON_CALL(dtype_obj.AsString(&found_dtype));
    if (found_dtype != "TYPE_INVALID") {
      if (dtype != DALI_NO_TYPE) {
        if (found_dtype != to_triton_config(dtype)) {
          throw TritonError::InvalidArg(make_string(
            "Mismatch of data_type config for \"", name, "\".\n"
            "Data type defined in config: ", found_dtype, "\n"
            "Data type defined in pipeline: ", to_triton_config(dtype)));
        }
      }
      return found_dtype;
    }
  }
  if (!allow_missing) {
    throw TritonError::InvalidArg(make_string("Missing data_type config for \"", name, "\""));
  }
  return to_triton_config(dtype);
}


std::string AutofillDtypeConfig(TritonJson::Value &io_object, const std::string &name,
                                dali_data_type_t dtype) {
  return ProcessDtypeConfig<true>(io_object, name, dtype);
}


void ValidateDtypeConfig(TritonJson::Value &io_object, const std::string &name,
                        dali_data_type_t dtype) {
  ProcessDtypeConfig<false>(io_object, name, dtype);
}


void AutofillShapeConfig(TritonJson::Value &config, TritonJson::Value &config_io,
                         const std::vector<int64_t> &model_io_shape) {
  std::string name;
  TRITON_CALL(config_io.MemberAsString("name", &name));
  TritonJson::Value config_dims_obj;
  if (config_io.Find("dims", &config_dims_obj)) {
    auto config_dims = ReadShape(config_dims_obj);
    if (config_dims.size() > 0) {
      auto new_dims = MatchShapes(name, config_dims, model_io_shape);
      SetShapeArray(config_dims_obj, new_dims);
    } else {
      SetShapeArray(config_dims_obj, model_io_shape);
    }
  } else {
    TritonJson::Value new_dims_obj(config, TritonJson::ValueType::ARRAY);
    SetShapeArray(new_dims_obj, model_io_shape);
    config_io.Add("dims", std::move(new_dims_obj));
  }
}


void ValidateShapeConfig(TritonJson::Value &io_object, const std::string &name,
                         const std::optional<std::vector<int64_t>> &shape) {
  TritonJson::Value dims_obj;
  TritonError error{io_object.MemberAsArray("dims", &dims_obj)};
  if (error) {
    throw TritonError::InvalidArg(make_string("Missing dims config for \"", name, "\""));
  }

  if (shape) {
    auto config_shape = ReadShape(dims_obj);
    MatchShapes(name, config_shape, *shape);
  }

}


void AutofillIOConfig(TritonJson::Value &config, TritonJson::Value &config_io,
                      const IOConfig &model_io) {
  TRITON_CALL(config_io.AssertType(common::TritonJson::ValueType::OBJECT));

  if (!config_io.Find("name")) {
    config_io.AddString("name", model_io.name);
  }

  std::string new_data_type = AutofillDtypeConfig(config_io, model_io.name, model_io.dtype);
  TritonJson::Value config_data_type;
  if (config_io.Find("data_type", &config_data_type)) {
    config_data_type.SetString(new_data_type);
  } else {
    config_io.AddString("data_type", new_data_type);
  }

  if (model_io.shape) {
    AutofillShapeConfig(config, config_io, *model_io.shape);
  }
}


void ValidateIOConfig(TritonJson::Value &io_object, const IOConfig &io_config) {
  TRITON_CALL(io_object.AssertType(common::TritonJson::ValueType::OBJECT));
  std::string name;
  io_object.MemberAsString("name", &name);
  ValidateDtypeConfig(io_object, name, io_config.dtype);
  ValidateShapeConfig(io_object, name, io_config.shape);
}


void ValidateAgainstTooManyInputs(TritonJson::Value &ins, const std::vector<IOConfig> &in_configs) {
   for (size_t i = 0; i < ins.ArraySize(); ++i) {
    TritonJson::Value io_object(TritonJson::ValueType::OBJECT);
    ins.IndexAsObject(i, &io_object);
    std::string name;
    if (io_object.MemberAsString("name", &name) != TRITONJSON_STATUSSUCCESS) {
      throw TritonError::InvalidArg(
        make_string("The input at index ", i,
                    " in the model configuration does not contain a `name` field."));
    }

    bool in_present = std::any_of(in_configs.begin(), in_configs.end(),
                                  [&name](const auto &ioc) { return ioc.name == name; });
    if (!in_present) {
      throw TritonError::InvalidArg(make_string("Configuration file contains config for ", name,
                                                " but such input is not present in the pipeline."));
    }
  }
}


void AutofillInputsConfig(TritonJson::Value &config, TritonJson::Value &config_ins,
                          const std::vector<IOConfig> &model_ins) {
  TRITON_CALL(config_ins.AssertType(common::TritonJson::ValueType::ARRAY));
  ValidateAgainstTooManyInputs(config_ins, model_ins);
  for (const auto &model_in: model_ins) {
    TritonJson::Value config_in(config, TritonJson::ValueType::OBJECT);
    auto found = FindObjectByName(config_ins, model_in.name, &config_in);
    AutofillIOConfig(config, config_in, model_in);
    if (!config_in.Find("allow_ragged_batch")) {
      config_in.AddBool("allow_ragged_batch", true);
    }
    if (!found) {
      config_ins.Append(std::move(config_in));
    }
  }
}


void AutofillOutputsConfig(TritonJson::Value &config, TritonJson::Value &config_outs,
                           const std::vector<IOConfig> &model_outs) {
  TRITON_CALL(config_outs.AssertType(common::TritonJson::ValueType::ARRAY));
  if (config_outs.ArraySize() > model_outs.size()) {
    throw TritonError::InvalidArg(
      make_string("The number of outputs specified in the DALI pipeline and the configuration"
                  " file do not match."
                  "\nModel config outputs: ", config_outs.ArraySize(),
                  "\nPipeline outputs: ", model_outs.size()));
  }

  size_t i = 0;
  for (; i < config_outs.ArraySize(); ++i) {
    TritonJson::Value config_out;
    TRITON_CALL(config_outs.IndexAsObject(i, &config_out));
    AutofillIOConfig(config, config_out, model_outs[i]);
  }

  for (; i < model_outs.size(); ++i) {
    TritonJson::Value config_out(config, TritonJson::ValueType::OBJECT);
    AutofillIOConfig(config, config_out, model_outs[i]);
    config_outs.Append(std::move(config_out));
  }
}


void AutofillConfig(TritonJson::Value &config, const std::vector<IOConfig> &model_ins,
                    const std::vector<IOConfig> &model_outs, int model_max_batch_size) {
  TritonJson::Value config_ins;
  if (config.Find("input", &config_ins)) {
    AutofillInputsConfig(config, config_ins, model_ins);
  } else {
    config_ins = TritonJson::Value(config, TritonJson::ValueType::ARRAY);
    AutofillInputsConfig(config, config_ins, model_ins);
    config.Add("input", std::move(config_ins));
  }

  TritonJson::Value config_outs;
  if (config.Find("output", &config_outs)) {
    AutofillOutputsConfig(config, config_outs, model_outs);
  } else {
    config_outs = TritonJson::Value(config, TritonJson::ValueType::ARRAY);
    AutofillOutputsConfig(config, config_outs, model_outs);
    config.Add("output", std::move(config_outs));
  }

  TritonJson::Value config_max_bs;
  if (config.Find("max_batch_size", &config_max_bs)) {
    int64_t config_max_bs_int = -1;
    TritonError{config_max_bs.AsInt(&config_max_bs_int)};  // immediately release error
    if (config_max_bs_int <= 0) {
      config_max_bs.SetInt(model_max_batch_size);
    }
  } else {
    config.AddInt("max_batch_size", model_max_batch_size);
  }
}


void ValidateInputs(TritonJson::Value &ins, const std::vector<IOConfig> &in_configs) {
  TRITON_CALL(ins.AssertType(common::TritonJson::ValueType::ARRAY));
  ValidateAgainstTooManyInputs(ins, in_configs);
  for (const auto &in_config: in_configs) {
    TritonJson::Value in_object(TritonJson::ValueType::OBJECT);
    auto ind = FindObjectByName(ins, in_config.name, &in_object);
    if (!ind) {
      throw TritonError::InvalidArg(
        make_string("Missing config for \"", in_config.name, "\" input."));
    }
    ValidateIOConfig(in_object, in_config);
  }
}


void ValidateOutputs(TritonJson::Value &outs, const std::vector<IOConfig> &out_configs) {
  TRITON_CALL(outs.AssertType(common::TritonJson::ValueType::ARRAY));
  if (outs.ArraySize() != out_configs.size()) {
    throw TritonError::InvalidArg(
      make_string("The number of outputs specified in the DALI pipeline and the "
                  "configuration file do not match."
                  "\nModel config outputs: ", outs.ArraySize(),
                  "\nPipeline outputs: ", out_configs.size()));
  }
  for (size_t i = 0; i < out_configs.size(); ++i) {
    TritonJson::Value out_object;
    TRITON_CALL(outs.IndexAsObject(i, &out_object));
    std::string name;
    if (out_object.MemberAsString("name", &name) != TRITONJSON_STATUSSUCCESS) {
      throw TritonError::InvalidArg(
        make_string("The output at index ", i,
                    " in the model configuration does not contain a `name` field."));
    }
    ValidateIOConfig(out_object, out_configs[i]);
  }
}


int ReadMaxBatchSize(TritonJson::Value &config) {
  int64_t bs = -1;
  TritonError{config.MemberAsInt("max_batch_size", &bs)};  // immediately release error
  if (bs > std::numeric_limits<int>::max() || bs < -1) {
    throw TritonError::InvalidArg(
      make_string("Invalid value of max_batch_size in model configuration: ", bs));
  }
  if (bs > 0) {
    return static_cast<int>(bs);
  } else {
    return -1;
  }
}


void ValidateConfig(TritonJson::Value &config, const std::vector<IOConfig> &in_configs,
                    const std::vector<IOConfig> &out_configs) {
  if (ReadMaxBatchSize(config) < 1) {
    throw TritonError::InvalidArg("Missing max_batch_size field in model configuration.");
  }

  TritonError err;

  TritonJson::Value inputs(config, TritonJson::ValueType::ARRAY);
  err = config.MemberAsArray("input", &inputs);
  if (err) {
    throw TritonError::InvalidArg("Missing inputs config.");
  }
  ValidateInputs(inputs, in_configs);

  TritonJson::Value outputs(config, TritonJson::ValueType::ARRAY);
  err = config.MemberAsArray("output", &outputs);
  if (err) {
    throw TritonError::InvalidArg("Missing outputs config.");
  }
  ValidateOutputs(outputs, out_configs);
}
}}}  // namespace triton::backend::dali
