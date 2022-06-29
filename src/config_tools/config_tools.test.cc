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


#include <catch2/catch.hpp>

#include "src/config_tools/config_tools.h"

namespace triton { namespace backend { namespace dali { namespace test {

using Catch::Matchers::Contains;

static void CheckIOConfigEquals(TritonJson::Value &io, IOConfig io_config) {
  CHECK(io.AssertType(TritonJson::ValueType::OBJECT) == TRITONJSON_STATUSSUCCESS);

  std::string name;
  CHECK(io.MemberAsString("name", &name) == TRITONJSON_STATUSSUCCESS);
  CHECK(name == io_config.name);

  TritonJson::Value dims(TritonJson::ValueType::ARRAY);
  if (!io_config.shape) {
    if (io.MemberAsArray("dims", &dims) == TRITONJSON_STATUSSUCCESS) {
      CHECK(dims.ArraySize() == 0);
    }
  } else {
    REQUIRE(io.MemberAsArray("dims", &dims) == TRITONJSON_STATUSSUCCESS);
    CHECK(ReadShape(dims) == *io_config.shape);
  }

  std::string data_type;
  if (io_config.dtype == DALI_NO_TYPE) {
    if (io.MemberAsString("data_type", &data_type) == TRITONJSON_STATUSSUCCESS) {
      CHECK(data_type == "TYPE_INVALID");
    }
  } else {
    REQUIRE(io.MemberAsString("data_type", &data_type) == TRITONJSON_STATUSSUCCESS);
    CHECK(data_type == to_triton_config(io_config.dtype));
  }
}

TEST_CASE("ReadShape test") {
  SECTION("Empty") {
    TritonJson::Value dims;
    TRITON_CALL(dims.Parse(R"json([])json"));
    CHECK(ReadShape(dims) == std::vector<int64_t>{});
  }

  SECTION("Single-dim") {
    TritonJson::Value dims;
    TRITON_CALL(dims.Parse(R"json([-1])json"));
    CHECK(ReadShape(dims) == std::vector<int64_t>{-1});
  }

  SECTION("Multi-dim") {
    TritonJson::Value dims;
    TRITON_CALL(dims.Parse(R"json([3, 2, 1])json"));
    CHECK(ReadShape(dims) == std::vector<int64_t>{3, 2, 1});
  }
}

TEST_CASE("IO config validation") {
  TritonJson::Value io_config;
  TRITON_CALL(io_config.Parse(R"json({
    "name": "io0",
    "dims": [3, 2, 1],
    "data_type": "TYPE_FP32"
  })json"));

  SECTION("Matching config") {
    ValidateIOConfig(io_config, IOConfig("io0", DALI_FLOAT, {{3, 2, 1}}));
    ValidateIOConfig(io_config, IOConfig("io0", DALI_NO_TYPE, {{3, 2, 1}}));
    ValidateIOConfig(io_config, IOConfig("io0", DALI_FLOAT, {}));
  }

  SECTION("Mismatching dtype") {
    REQUIRE_THROWS_WITH(
      ValidateIOConfig(io_config, IOConfig("io0", DALI_INT32, {{3, 2, 1}})),
      Contains("Data type defined in config: TYPE_FP32") &&
      Contains("Data type defined in pipeline: TYPE_INT32"));
  }

  SECTION("Mismatching ndims") {
    REQUIRE_THROWS_WITH(
      ValidateIOConfig(io_config, IOConfig("io0", DALI_FLOAT, {{-1, -1, -1, -1}})),
      Contains("Number of dimensions defined in config: 3") &&
      Contains("Number of dimensions defined in pipeline: 4"));
  }

  SECTION("Mismatching shapes") {
    REQUIRE_THROWS_WITH(
      ValidateIOConfig(io_config, IOConfig("io0", DALI_FLOAT, {{3, 2, 2}})),
      Contains("Dims defined in config: {3, 2, 1}") &&
      Contains("Dims defined in pipeline: {3, 2, 2}"));
  }
}

TEST_CASE("IOs config validation") {
  TritonJson::Value ios;
  TRITON_CALL(ios.Parse(R"json([
    {
      "name": "io1",
      "dims": [3, 2, 1],
      "data_type": "TYPE_FP32"
    },
    {
      "name": "io0",
      "dims": [1, 2, -1],
      "data_type": "TYPE_FP16"
    }
  ])json"));

  SECTION("Correct config") {
    ValidateIOsConfig(ios, {IOConfig("io0"), IOConfig("io1")});
  }

  SECTION("Missing config") {
    REQUIRE_THROWS_WITH(
      (ValidateIOsConfig(ios, {IOConfig("io0"), IOConfig("io1"), IOConfig("io3")})),
      Contains("Missing config for \"io3\""));
  }
}

TEST_CASE("IO auto config") {
  IOConfig io_config_full("io", DALI_FLOAT, {{-1, -1, 3}});
  IOConfig io_config_notype("io", DALI_NO_TYPE, {{-1, -1, 3}});
  IOConfig io_config_noshape("io", DALI_FLOAT, {});

  TritonJson::Value io_empty;
  TRITON_CALL(io_empty.Parse(R"json({})json"));

  TritonJson::Value io_full;
  TRITON_CALL(io_full.Parse(R"json({
    "name": "io",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32"
  })json"));

  TritonJson::Value io_notype;
  TRITON_CALL(io_notype.Parse(R"json({
    "name": "io",
    "dims": [3, 2, 3]
  })json"));

  TritonJson::Value io_noshape;
  TRITON_CALL(io_noshape.Parse(R"json({
    "name": "io",
    "data_type": "TYPE_FP32"
  })json"));

  SECTION("Full auto-config") {
    TritonJson::Value result(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_empty, io_config_full, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{-1, -1, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_full, io_config_full, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_notype, io_config_full, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_noshape, io_config_full, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{-1, -1, 3}}));
  }

  SECTION("No-type auto-config") {
    TritonJson::Value result(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_empty, io_config_notype, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_NO_TYPE, {{-1, -1, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_full, io_config_notype, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_notype, io_config_notype, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_NO_TYPE, {{3, 2, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_noshape, io_config_notype, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{-1, -1, 3}}));
  }

  SECTION("No-shape auto-config") {
    TritonJson::Value result(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_empty, io_config_noshape, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_full, io_config_noshape, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_notype, io_config_noshape, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    result = TritonJson::Value(TritonJson::ValueType::OBJECT);
    AutofillIOConfig(io_noshape, io_config_noshape, result);
    CheckIOConfigEquals(result, IOConfig("io", DALI_FLOAT, {}));
  }
}

TEST_CASE("IOs validation") {
  TritonJson::Value ios(TritonJson::ValueType::ARRAY);
  TRITON_CALL(ios.Parse(R"json([
  {
    "name": "io1",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32"
  },
  {
    "name": "io2",
    "dims": [1, 1],
    "data_type": "TYPE_FP16"
  }
  ])json"));

  SECTION("Correct config") {
    std::vector<IOConfig> ios_config = {
      IOConfig("io1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("io2", DALI_FLOAT16, {{1, 1}})
    };
    ValidateIOsConfig(ios, ios_config);
  }

  SECTION("Missing input") {
    std::vector<IOConfig> ios_config = {
      IOConfig("io1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("io2", DALI_FLOAT16, {{1, 1}}),
      IOConfig("io3", DALI_UINT16, {{1}})
    };

    REQUIRE_THROWS_WITH(ValidateIOsConfig(ios, ios_config),
                        Contains("Missing config for \"io3\""));
  }
}

TEST_CASE("IOs auto-config") {
  TritonJson::Value ios(TritonJson::ValueType::ARRAY);
  TRITON_CALL(ios.Parse(R"json([
  {
    "name": "io1",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32",
    "allow_ragged_batches": true
  },
  {
    "name": "io2",
    "dims": [5, 5]
  }
  ])json"));

  std::vector<IOConfig> ios_config = {
    IOConfig("io1", DALI_FLOAT, {{3, 2, 3}}),
    IOConfig("io2", DALI_FLOAT16, {{5, 5}}),
    IOConfig("io3", DALI_UINT16, {{4}})
  };

  SECTION("Inputs auto-config") {
    TritonJson::Value inputs_conf(TritonJson::ValueType::ARRAY);
    AutofillInputsConfig(ios, ios_config, inputs_conf);

    for (auto &config: ios_config) {
      TritonJson::Value inp_object;
      REQUIRE(FindObjectByName(inputs_conf, config.name, &inp_object));
      bool ragged_batches;
      REQUIRE(
        inp_object.MemberAsBool("allow_ragged_batches", &ragged_batches) == TRITONJSON_STATUSSUCCESS
      );
      REQUIRE(ragged_batches);
      CheckIOConfigEquals(inp_object, config);
    }
  }

  SECTION("Outputs auto-config") {
    TritonJson::Value outputs_conf(TritonJson::ValueType::ARRAY);
    AutofillInputsConfig(ios, ios_config, outputs_conf);

    for (auto &config: ios_config) {
      TritonJson::Value out_object;
      REQUIRE(FindObjectByName(outputs_conf, config.name, &out_object));
      CheckIOConfigEquals(out_object, config);
    }
  }
}

}}}}  // namespace triton::backend::dali::test
