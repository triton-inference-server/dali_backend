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

static void CheckIOConfigEquals(TritonJson::Value &io, IOConfig io_config, bool compare_names = true) {
  CHECK(io.AssertType(TritonJson::ValueType::OBJECT) == TRITONJSON_STATUSSUCCESS);

  if (compare_names) {
    std::string name;
    CHECK(io.MemberAsString("name", &name) == TRITONJSON_STATUSSUCCESS);
    CHECK(name == io_config.name);
  }

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


TEST_CASE("Inputs validation") {
  TritonJson::Value ios(TritonJson::ValueType::ARRAY);
  TRITON_CALL(ios.Parse(R"json([
  {
    "name": "i1",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32"
  },
  {
    "name": "i2",
    "dims": [1, 1],
    "data_type": "TYPE_FP16"
  }
  ])json"));

  SECTION("Correct config") {
    std::vector<IOConfig> ios_config = {
      IOConfig("i1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("i2", DALI_FLOAT16, {{1, 1}})
    };
    ValidateInputs(ios, ios_config);
  }

  SECTION("Missing input") {
    std::vector<IOConfig> ios_config = {
      IOConfig("i1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("i2", DALI_FLOAT16, {{1, 1}}),
      IOConfig("i3", DALI_UINT16, {{1}})
    };

    REQUIRE_THROWS_WITH(ValidateInputs(ios, ios_config),
                        Contains("Missing config for \"i3\""));
  }
}


TEST_CASE("Outputs validation") {
  TritonJson::Value ios(TritonJson::ValueType::ARRAY);
  TRITON_CALL(ios.Parse(R"json([
  {
    "name": "o1",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32"
  },
  {
    "name": "o2",
    "dims": [1, 1],
    "data_type": "TYPE_FP16"
  }
  ])json"));

  SECTION("Correct config") {
    std::vector<IOConfig> ios_config = {
      IOConfig("Pipe_o1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("Pipe_o2", DALI_FLOAT16, {{1, 1}})
    };
    ValidateOutputs(ios, ios_config);
  }

  SECTION("Missing output") {
    std::vector<IOConfig> ios_config = {
      IOConfig("Pipe_o1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("Pipe_o2", DALI_FLOAT16, {{1, 1}}),
      IOConfig("Pipe_o3", DALI_UINT16, {{1}})
    };

    REQUIRE_THROWS_WITH(ValidateOutputs(ios, ios_config),
                        Contains("The number of outputs specified in the DALI pipeline and the"
                                 " configuration file do not match."
                                 "\nModel config outputs: 2"
                                 "\nPipeline outputs: 3"));
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
    AutofillIOConfig(io_empty, io_empty, io_config_full);
    CheckIOConfigEquals(io_empty, IOConfig("io", DALI_FLOAT, {{-1, -1, 3}}));

    AutofillIOConfig(io_full, io_full, io_config_full);
    CheckIOConfigEquals(io_full, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    AutofillIOConfig(io_notype, io_notype, io_config_full);
    CheckIOConfigEquals(io_notype, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    AutofillIOConfig(io_noshape, io_noshape, io_config_full);
    CheckIOConfigEquals(io_noshape, IOConfig("io", DALI_FLOAT, {{-1, -1, 3}}));
  }

  SECTION("No-type auto-config") {
    AutofillIOConfig(io_empty, io_empty, io_config_notype);
    CheckIOConfigEquals(io_empty, IOConfig("io", DALI_NO_TYPE, {{-1, -1, 3}}));

    AutofillIOConfig(io_full, io_full, io_config_notype);
    CheckIOConfigEquals(io_full, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    AutofillIOConfig(io_notype, io_notype, io_config_notype);
    CheckIOConfigEquals(io_notype, IOConfig("io", DALI_NO_TYPE, {{3, 2, 3}}));

    AutofillIOConfig(io_noshape, io_noshape, io_config_notype);
    CheckIOConfigEquals(io_noshape, IOConfig("io", DALI_FLOAT, {{-1, -1, 3}}));
  }

  SECTION("No-shape auto-config") {
    AutofillIOConfig(io_empty, io_empty, io_config_noshape);
    CheckIOConfigEquals(io_empty, IOConfig("io", DALI_FLOAT, {}));

    AutofillIOConfig(io_full, io_full, io_config_noshape);
    CheckIOConfigEquals(io_full, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    AutofillIOConfig(io_notype, io_notype, io_config_noshape);
    CheckIOConfigEquals(io_notype, IOConfig("io", DALI_FLOAT, {{3, 2, 3}}));

    AutofillIOConfig(io_noshape, io_noshape, io_config_noshape);
    CheckIOConfigEquals(io_noshape, IOConfig("io", DALI_FLOAT, {}));
  }
}


TEST_CASE("Inputs auto-config") {
  TritonJson::Value ios(TritonJson::ValueType::ARRAY);
  TRITON_CALL(ios.Parse(R"json([
  {
    "name": "i1",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32",
    "allow_ragged_batch": true
  },
  {
    "name": "i2",
    "dims": [5, 5]
  }
  ])json"));

  SECTION("Inputs auto-config") {
    std::vector<IOConfig> model_ins = {
      IOConfig("i1", DALI_FLOAT, {{3, 2, 3}}),
      IOConfig("i2", DALI_FLOAT16, {{5, 5}}),
      IOConfig("i3", DALI_UINT16, {{4}})
    };

    AutofillInputsConfig(ios, ios, model_ins);

    for (auto &model_in: model_ins) {
      TritonJson::Value inp_object;
      REQUIRE(FindObjectByName(ios, model_in.name, &inp_object));
      bool ragged_batches;
      REQUIRE(
        inp_object.MemberAsBool("allow_ragged_batch", &ragged_batches) == TRITONJSON_STATUSSUCCESS
      );
      REQUIRE(ragged_batches);
      CheckIOConfigEquals(inp_object, model_in);
    }
  }


  SECTION("Inputs auto-config, reordered") {
    std::vector<IOConfig> model_ins = {
      IOConfig("i0", DALI_INT32, {{-1, -1}}),
      IOConfig("i2", DALI_FLOAT16, {{5, 5}}),
      IOConfig("i1", DALI_FLOAT, {{3, 2, 3}}),
    };

    AutofillInputsConfig(ios, ios, model_ins);

    // New config keeps the order of inputs from the original config
    TritonJson::Value inp_object;
    REQUIRE(ios.IndexAsObject(0, &inp_object) == TRITONJSON_STATUSSUCCESS);
    bool ragged_batches;
    REQUIRE(
      inp_object.MemberAsBool("allow_ragged_batch", &ragged_batches) == TRITONJSON_STATUSSUCCESS
    );
    REQUIRE(ragged_batches);
    CheckIOConfigEquals(inp_object, IOConfig("i1", DALI_FLOAT, {{3, 2, 3}}));

    REQUIRE(ios.IndexAsObject(1, &inp_object) == TRITONJSON_STATUSSUCCESS);
    REQUIRE(
      inp_object.MemberAsBool("allow_ragged_batch", &ragged_batches) == TRITONJSON_STATUSSUCCESS
    );
    REQUIRE(ragged_batches);
    CheckIOConfigEquals(inp_object, IOConfig("i2", DALI_FLOAT16, {{5, 5}}));

    REQUIRE(ios.IndexAsObject(2, &inp_object) == TRITONJSON_STATUSSUCCESS);
    REQUIRE(
      inp_object.MemberAsBool("allow_ragged_batch", &ragged_batches) == TRITONJSON_STATUSSUCCESS
    );
    REQUIRE(ragged_batches);
    CheckIOConfigEquals(inp_object, IOConfig("i0", DALI_INT32, {{-1, -1}}));
  }
}


TEST_CASE("Outputs auto-config") {
  TritonJson::Value outs(TritonJson::ValueType::ARRAY);
  TRITON_CALL(outs.Parse(R"json([
  {
    "name": "o1",
    "dims": [3, 2, 3],
    "data_type": "TYPE_FP32"
  },
  {
    "name": "o2",
    "dims": [5, 5]
  }
  ])json"));

  std::vector<IOConfig> model_outs = {
    IOConfig("Pipe_o1", DALI_FLOAT, {{3, 2, 3}}),
    IOConfig("Pipe_o2", DALI_FLOAT16, {{5, 5}}),
    IOConfig("Pipe_o3", DALI_UINT16, {{4}})
  };


  SECTION("Outputs auto-config") {
    AutofillOutputsConfig(outs, outs, model_outs);

    // In case of outputs, names from the config file take precedence and
    // override the names coming from the pipeline
    std::vector<std::string> names_order = {"o1", "o2", "Pipe_o3"};
    REQUIRE(outs.ArraySize() == 3);
    for (size_t i = 0; i < outs.ArraySize(); ++i) {
      TritonJson::Value out_object;
      REQUIRE(outs.IndexAsObject(i, &out_object) == TRITONJSON_STATUSSUCCESS);
      std::string name;
      REQUIRE(out_object.MemberAsString("name", &name) == TRITONJSON_STATUSSUCCESS);
      REQUIRE(name == names_order[i]);
      CheckIOConfigEquals(out_object, model_outs[i], false);
    }
  }
}


TEST_CASE("Autofill config") {
  TritonJson::Value config(TritonJson::ValueType::OBJECT);
  TRITON_CALL(config.Parse(R"json({
    "input": [
      {
        "name": "i1",
        "dims": [3, 2, 1],
        "data_type": "TYPE_FP16"
      },
      {
        "name": "i2",
        "dims": [-1, -1, 3],
        "data_type": "TYPE_FP32"
      }
    ],
    "output": [
      {
        "name": "o1",
        "dims": [-1, 2, 3],
        "data_type": "TYPE_FP32"
      }
    ]
  })json"));

  std::vector<IOConfig> model_ins = {
    IOConfig("i1", DALI_FLOAT16, {{3, 2, 1}}),
    IOConfig("i2", DALI_NO_TYPE, {{-1, 3, 3}}),
    IOConfig("i3", DALI_INT32, {{1, 1, 1}})
  };

  std::vector<IOConfig> model_outs = {
    IOConfig("Pipe_o1", DALI_FLOAT, {{3, 2, 3}}),
    IOConfig("o2", DALI_INT32, {{-1, -1}})
  };

  std::string expected_config = R"json({
    "input": [
        {
            "name": "i1",
            "dims": [
                3,
                2,
                1
            ],
            "data_type": "TYPE_FP16",
            "allow_ragged_batch": true
        },
        {
            "name": "i2",
            "dims": [
                -1,
                3,
                3
            ],
            "data_type": "TYPE_FP32",
            "allow_ragged_batch": true
        },
        {
            "name": "i3",
            "data_type": "TYPE_INT32",
            "dims": [
                1,
                1,
                1
            ],
            "allow_ragged_batch": true
        }
    ],
    "output": [
        {
            "name": "o1",
            "dims": [
                3,
                2,
                3
            ],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "o2",
            "data_type": "TYPE_INT32",
            "dims": [
                -1,
                -1
            ]
        }
    ],
    "max_batch_size": 13
})json";

  AutofillConfig(config, model_ins, model_outs, 13);
  common::TritonJson::WriteBuffer buffer;
  config.PrettyWrite(&buffer);
  REQUIRE(buffer.Contents() == expected_config);
}


TEST_CASE("Read max_batch_size") {
  SECTION("correct bs") {
    TritonJson::Value config(TritonJson::ValueType::OBJECT);
    TRITON_CALL(config.Parse(R"json({
    "max_batch_size": 32,

    "output": [
      {
        "name": "o1",
        "dims": [3, 2, 3],
        "data_type": "TYPE_FP32"
      },
      {
        "name": "o2",
        "dims": [5, 5]
      }
    ]
    })json"));

    REQUIRE(ReadMaxBatchSize(config) == 32);
  }

  SECTION("incorrect bs") {
    TritonJson::Value config(TritonJson::ValueType::OBJECT);
    TRITON_CALL(config.Parse(R"json({
    "max_batch_size": -2,
    "input": [
      {
        "name": "i1",
        "dims": [3, 2, 3],
        "data_type": "TYPE_FP32",
        "allow_ragged_batch": true
      },
      {
        "name": "i2",
        "dims": [5, 5]
      }
    ],
    "output": [
      {
        "name": "o1",
        "dims": [3, 2, 3],
        "data_type": "TYPE_FP32"
      },
      {
        "name": "o2",
        "dims": [5, 5]
      }
    ]
    })json"));

    REQUIRE_THROWS_WITH(ReadMaxBatchSize(config),
                        Contains("Invalid value of max_batch_size in model configuration: -2"));
  }
}


TEST_CASE("Validate config") {
  std::vector<IOConfig> ins_config = {
    IOConfig("i1", DALI_FLOAT16, {{3, 2, 1}})
  };

  std::vector<IOConfig> outs_config = {
    IOConfig("Pipe_o1", DALI_FLOAT, {{3, 2, 3}})
  };

  SECTION("correct config") {
    TritonJson::Value config(TritonJson::ValueType::OBJECT);
    TRITON_CALL(config.Parse(R"json({
      "max_batch_size": 1,
      "input": [
        {
          "name": "i1",
          "dims": [3, 2, 1],
          "data_type": "TYPE_FP16",
          "allow_ragged_batch": true
        }
      ],
      "output": [
        {
          "name": "o1",
          "dims": [3, 2, 3],
          "data_type": "TYPE_FP32"
        }
      ]
    })json"));

    ValidateConfig(config, ins_config, outs_config);
  }

  SECTION("missing inputs") {
    TritonJson::Value config(TritonJson::ValueType::OBJECT);
    TRITON_CALL(config.Parse(R"json({
      "max_batch_size": 1,
      "output": [
        {
          "name": "o1",
          "dims": [3, 2, 3],
          "data_type": "TYPE_FP32"
        }
      ]
    })json"));

    REQUIRE_THROWS_WITH(ValidateConfig(config, ins_config, outs_config),
                        Contains("Missing inputs config."));
  }

  SECTION("missing outputs") {
    TritonJson::Value config(TritonJson::ValueType::OBJECT);
    TRITON_CALL(config.Parse(R"json({
      "max_batch_size": 1,
      "input": [
        {
          "name": "i1",
          "dims": [3, 2, 1],
          "data_type": "TYPE_FP16",
          "allow_ragged_batch": true
        }
      ]
    })json"));

    REQUIRE_THROWS_WITH(ValidateConfig(config, ins_config, outs_config),
                        Contains("Missing outputs config."));
  }

  SECTION("missing max_batch_size") {
    TritonJson::Value config(TritonJson::ValueType::OBJECT);
    TRITON_CALL(config.Parse(R"json({
      "input": [
        {
          "name": "i1",
          "dims": [3, 2, 1],
          "data_type": "TYPE_FP16",
          "allow_ragged_batch": true
        }
      ],
      "output": [
        {
          "name": "o1",
          "dims": [3, 2, 3],
          "data_type": "TYPE_FP32"
        }
      ]
    })json"));

    REQUIRE_THROWS_WITH(ValidateConfig(config, ins_config, outs_config),
                        Contains("Missing max_batch_size field in model configuration."));
  }
}

}}}}  // namespace triton::backend::dali::test
