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


template <bool allow_missing>
void ProcessShapeConfig(TritonJson::Value &io_object, const std::string &name,
                        const std::optional<std::vector<int64_t>> &shape,
                        TritonJson::Value &resulting_dims) {
  TritonJson::Value dims_obj;
  TritonError error{io_object.MemberAsArray("dims", &dims_obj)};
  if (!error) {
    auto config_shape = ReadShape(dims_obj);
    if (shape) {
      auto resulting_shape = MatchShapes(name, config_shape, *shape);
      SetShapeArray(resulting_dims, resulting_shape);
    } else {
      SetShapeArray(resulting_dims, config_shape);
    }
  } else if (allow_missing) {
    if (shape) {
      SetShapeArray(resulting_dims, *shape);
    } else {
      SetShapeArray(resulting_dims, {});
    }
  } else {
    throw TritonError::InvalidArg(make_string("Missing dims config for \"", name, "\""));
  }
}


void AutofillShapeConfig(TritonJson::Value &io_object, const std::string &name,
                         const std::optional<std::vector<int64_t>> &shape,
                         TritonJson::Value &resulting_dims) {
  ProcessShapeConfig<true>(io_object, name, shape, resulting_dims);
}


void ValidateShapeConfig(TritonJson::Value &io_object, const std::string &name,
                         const std::optional<std::vector<int64_t>> &shape) {
  TritonJson::Value dummy(TritonJson::ValueType::ARRAY);
  ProcessShapeConfig<false>(io_object, name, shape, dummy);
}


void AutofillIOConfig(TritonJson::Value &io_object, const IOConfig &io_config,
                      TritonJson::Value &new_io_object) {
  TRITON_CALL(io_object.AssertType(common::TritonJson::ValueType::OBJECT));
  TRITON_CALL(new_io_object.AssertType(common::TritonJson::ValueType::OBJECT));

  std::string name;
  TritonError error{io_object.MemberAsString("name", &name)};
  if (!error) {
    new_io_object.AddString("name", name);
  } else {
    new_io_object.AddString("name", io_config.name);
  }

  std::string new_data_type = AutofillDtypeConfig(io_object, io_config.name, io_config.dtype);
  new_io_object.AddString("data_type", new_data_type);

  TritonJson::Value new_dims;
  new_io_object.Add("dims", TritonJson::Value(new_io_object, TritonJson::ValueType::ARRAY));
  new_io_object.MemberAsArray("dims", &new_dims);
  AutofillShapeConfig(io_object, io_config.name, io_config.shape, new_dims);
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


void AutofillInputsConfig(TritonJson::Value &ins, const std::vector<IOConfig> &in_configs,
                          TritonJson::Value &new_ins) {
  TRITON_CALL(ins.AssertType(common::TritonJson::ValueType::ARRAY));
  TRITON_CALL(new_ins.AssertType(common::TritonJson::ValueType::ARRAY));
  ValidateAgainstTooManyInputs(ins, in_configs);
  std::vector<TritonJson::Value> new_in_objs(in_configs.size());
  auto end_ind = ins.ArraySize();
  for (const auto &in_config: in_configs) {
    TritonJson::Value in_object(TritonJson::ValueType::OBJECT);
    auto ind = FindObjectByName(ins, in_config.name, &in_object);
    size_t in_index;
    if (ind) {
      in_index = *ind;
    } else {
      in_index = end_ind++;
    }
    new_in_objs[in_index] = TritonJson::Value(new_ins, TritonJson::ValueType::OBJECT);
    AutofillIOConfig(in_object, in_config, new_in_objs[in_index]);

    // std::string new_name;
    // if (new_in_objs[in_index].MemberAsString("name", &new_name)) {
    //   std::cout << "has name: " << new_name << std::endl;
    // } else {
    //   std::cout << "no name" << std::endl;
    // }

    bool ragged_batches;
    TritonError error{in_object.MemberAsBool("allow_ragged_batches", &ragged_batches)};
    if (!error) {
      new_in_objs[in_index].AddBool("allow_ragged_batches", ragged_batches);
    } else {
      new_in_objs[in_index].AddBool("allow_ragged_batches", true);
    }
  }

  for (auto &new_in : new_in_objs) {
    new_ins.Append(std::move(new_in));
  }
}


void AutofillOutputsConfig(TritonJson::Value &outs, const std::vector<IOConfig> &out_configs,
                           TritonJson::Value &new_outs) {
  TRITON_CALL(outs.AssertType(common::TritonJson::ValueType::ARRAY));
  TRITON_CALL(new_outs.AssertType(common::TritonJson::ValueType::ARRAY));
  if (outs.ArraySize() > out_configs.size()) {
    throw TritonError::InvalidArg(
      make_string("The number of outputs specified in the DALI pipeline and the configuration"
                  " file do not match."
                  "\nModel config outputs: ", outs.ArraySize(),
                  "\nPipeline outputs: ", out_configs.size()));
  }

  std::vector<TritonJson::Value> new_out_objs(out_configs.size());
  for (size_t i = 0; i < out_configs.size(); ++i) {
    TritonJson::Value out_object(TritonJson::ValueType::OBJECT);
    TritonError{outs.IndexAsObject(i, &out_object)};
    new_out_objs[i] = TritonJson::Value(new_outs, TritonJson::ValueType::OBJECT);
    AutofillIOConfig(out_object, out_configs[i], new_out_objs[i]);
  }

  for (auto &new_out : new_out_objs) {
    new_outs.Append(std::move(new_out));
  }
}


void AutofillConfig(TritonJson::Value &config, const std::vector<IOConfig> &in_configs,
                    const std::vector<IOConfig> &out_configs, int pipeline_max_batch_size) {
  TritonJson::Value existing_ins;
  bool found_inputs = config.Find("input", &existing_ins);

  TritonJson::Value new_ins(config, TritonJson::ValueType::ARRAY);
  AutofillInputsConfig(existing_ins, in_configs, new_ins);

  if (found_inputs) {
    config.Remove("input");
  }
  config.Add("input", std::move(new_ins));

  TritonJson::Value existing_outs;
  bool found_outputs = config.Find("output", &existing_outs);

  TritonJson::Value new_outs(config, TritonJson::ValueType::ARRAY);
  AutofillOutputsConfig(existing_outs, out_configs, new_outs);

  // if (found_outputs) {
  //   config.Remove("output");
  // }
  // config.Add("output", std::move(new_outs));
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
  TritonError{config.MemberAsInt("max_batch_size", &bs)};
  if (bs > std::numeric_limits<int>::max() || bs < -1) {
    throw TritonError::InvalidArg(
      make_string("Invalid value of max_batch_size in model configuration: ", bs));
  }
  return static_cast<int>(bs);
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
