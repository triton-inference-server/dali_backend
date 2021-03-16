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

#include <algorithm>
#include <catch2/catch.hpp>

#include "src/dali_executor/dali_executor.h"
#include "src/dali_executor/test/test_utils.h"
#include "src/dali_executor/test_data.h"

namespace triton { namespace backend { namespace dali { namespace test {

TEST_CASE("Scaling Pipeline")
{
  std::string pipeline(
      (const char*)pipelines::scale_pipeline_str,
      pipelines::scale_pipeline_len);
  DaliExecutor executor(pipeline, 8, 4, 0);
  std::mt19937 rand(1217);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const std::string inp_name = "INPUT0";
  auto scaling_test = [&](int batch_size) {
    TensorListShape<> shape(batch_size, 2);
    for (int i = 0; i < batch_size; ++i) {
      shape.set_tensor_shape(i, TensorShape<>(i + 1, 50));
    }
    std::vector<float> input_buffer;
    auto input = RandomInput(
        input_buffer, inp_name, shape, [&]() { return dist(rand); });
    std::vector<IODescr<false>> input_vec;
    input_vec.emplace_back(std::move(input));
    auto output = executor.Run(input_vec);
    REQUIRE(shape == output[0].shape);
    std::vector<float> output_buffer(input_buffer.size());
    std::vector<IODescr<false>> output_vec(1);
    auto& outdesc = output_vec[0];
    outdesc.device = device_type_t::CPU;
    outdesc.buffer = make_span(
        reinterpret_cast<char*>(output_buffer.data()),
        output_buffer.size() * sizeof(decltype(output_buffer)::size_type));
    executor.PutOutputs(output_vec);
    for (size_t i = 0; i < input_buffer.size(); ++i) {
      REQUIRE(output_buffer[i] == input_buffer[i] * 2);
    }
  };

  SECTION("Simple execute")
  {
    scaling_test(2);
    scaling_test(4);
  }

  SECTION("Repeat batch size")
  {
    scaling_test(3);
    scaling_test(3);
  }
}

TEST_CASE("RN50 pipeline")
{
  std::string pipeline(
      (const char*)pipelines::rn50_gpu_dali_chr,
      pipelines::rn50_gpu_dali_len);
  DaliExecutor executor(pipeline, 1, 3, 0);
  IODescr<false> input;
  input.name = "DALI_INPUT_0";
  input.type = dali_data_type_t::DALI_UINT8;
  input.shape = TensorListShape<1>(1);
  input.shape.set_tensor_shape(0, TensorShape<>(data::jpeg_image_len));
  input.buffer = span<char>((char*)(data::jpeg_image_str),
                            data::jpeg_image_len);
  input.device = device_type_t::CPU;

  auto execute_with_image = [&]() {
    const float expected_values[]
      = {-2.1179, -2.03571, -1.80444}; // 0 values after normalization
    const int output_c = 3, output_h = 224, output_w = 224;
    auto output = executor.Run(std::vector<IODescr<false>>({input}));
    REQUIRE(output[0].shape.tensor_shape(0) == TensorShape<3>(output_c, output_h, output_w));
    std::vector<float> output_buffer(output[0].shape.num_elements());
    std::vector<IODescr<false>> output_vec(1);
    auto& outdesc = output_vec[0];
    outdesc.device = device_type_t::CPU;
    outdesc.device_id = 0;
    outdesc.buffer = make_span(
        (char*)output_buffer.data(),
        output_buffer.size() * sizeof(decltype(output_buffer)::size_type));
    executor.PutOutputs(output_vec);
    for (int c = 0; c < output_c; ++c) {
      for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
          REQUIRE(output_buffer[x + (y + c * output_h) * output_w] == Approx(expected_values[c]));
        }
      }
    }
  };

  SECTION("Simple execute")
  {
    execute_with_image();
  }

  SECTION("Recover from error")
  {
    auto rand_inp_shape = TensorListShape<1>(1);
    rand_inp_shape.set_tensor_shape(0, TensorShape<>(1024));
    std::vector<uint8_t> rand_input_buffer;
    std::mt19937 rand(1217);
    std::uniform_int_distribution<short> dist(0, 255);
    auto rand_input = RandomInput(
        rand_input_buffer, input.name, rand_inp_shape, [&]() { return dist(rand); });
    REQUIRE_THROWS(
      executor.Run(std::vector<IODescr<false>>({rand_input}))
    );

    REQUIRE_NOTHROW(
      execute_with_image()
    );
  }
}

}}}}  // namespace triton::backend::dali::test
