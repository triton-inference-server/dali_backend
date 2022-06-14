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
    auto err = obj.MemberAsString("name", &found_name);
    if (!err && found_name == name) {
      ret->Release();
      array.IndexAsObject(i, ret);
      return {i};
    }
  }
  return {};
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
    throw TritonError::InvalidArg(make_string("Mismatch in number of dimensions for ", name, "\n"
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
            "Mismatch of data_type config for \"", name, "\". \n"
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
  if (io_object.MemberAsArray("dims", &dims_obj) == TRITONJSON_STATUSSUCCESS) {
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

  new_io_object.AddString("name", io_config.name);

  std::string new_data_type = AutofillDtypeConfig(io_object, io_config.name, io_config.dtype);
  new_io_object.AddString("data_type", new_data_type);

  TritonJson::Value new_dims;
  new_io_object.Add("dims", TritonJson::Value(new_io_object, TritonJson::ValueType::ARRAY));
  new_io_object.MemberAsArray("dims", &new_dims);
  AutofillShapeConfig(io_object, io_config.name, io_config.shape, new_dims);
}


void ValidateIOConfig(TritonJson::Value &io_object, const IOConfig &io_config) {
  TRITON_CALL(io_object.AssertType(common::TritonJson::ValueType::OBJECT));
  ValidateDtypeConfig(io_object, io_config.name, io_config.dtype);
  ValidateShapeConfig(io_object, io_config.name, io_config.shape);
}


template <bool input>
void AutofillIOsConfig(TritonJson::Value &ios, const std::vector<IOConfig> &io_configs,
                       TritonJson::Value &new_ios) {
  TRITON_CALL(ios.AssertType(common::TritonJson::ValueType::ARRAY));
  TRITON_CALL(new_ios.AssertType(common::TritonJson::ValueType::ARRAY));

  std::vector<TritonJson::Value> new_io_objs(io_configs.size());
  auto end_ind = ios.ArraySize();
  for (const auto &io_config: io_configs) {
    TritonJson::Value io_object(TritonJson::ValueType::OBJECT);
    auto ind = FindObjectByName(ios, io_config.name, &io_object);
    size_t io_index;
    if (ind) {
      io_index = *ind;
    } else {
      io_index = end_ind++;
    }
    new_io_objs[io_index] = TritonJson::Value(new_ios, TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_object, io_config, new_io_objs[io_index]);

    if (input) {
      bool ragged_batches;
      if (io_object.MemberAsBool("allow_ragged_batches", &ragged_batches) 
            == TRITONJSON_STATUSSUCCESS) {
        new_io_objs[io_index].AddBool("allow_ragged_batches", ragged_batches);
      } else {
        new_io_objs[io_index].AddBool("allow_ragged_batches", true);
      }
    }
  }

  for (auto &new_io : new_io_objs) {
    new_ios.Append(std::move(new_io));
  }
}


void AutofillInputsConfig(TritonJson::Value &inputs, const std::vector<IOConfig> &in_configs,
                          TritonJson::Value &new_ios) {
  AutofillIOsConfig<true>(inputs, in_configs, new_ios);
}


void AutofillOutputsConfig(TritonJson::Value &outputs, const std::vector<IOConfig> &out_configs,
                           TritonJson::Value &new_ios) {
  AutofillIOsConfig<false>(outputs, out_configs, new_ios);
}

void ValidateIOsConfig(TritonJson::Value &ios, const std::vector<IOConfig> &io_configs) {
  TRITON_CALL(ios.AssertType(common::TritonJson::ValueType::ARRAY));
  for (const auto &io_config: io_configs) {
    TritonJson::Value io_object(TritonJson::ValueType::OBJECT);
    auto ind = FindObjectByName(ios, io_config.name, &io_object);
    if (!ind) {
      throw TritonError::InvalidArg(make_string("Missing config for \"", io_config.name, "\""));
    }
    ValidateIOConfig(io_object, io_config);
  }
}
}}}  // namespace triton::backend::dali
