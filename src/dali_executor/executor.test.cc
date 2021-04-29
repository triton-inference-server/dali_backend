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

TEST_CASE("Scaling Pipeline") {
  std::string pipeline_s((const char *)pipelines::scale_pipeline_str,
                         pipelines::scale_pipeline_len);
  DaliPipeline pipeline(pipeline_s, 8, 4, 0);
  DaliExecutor executor(std::move(pipeline));
  std::mt19937 rand(1217);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const std::string inp_name = "INPUT0";
  auto scaling_test = [&](const std::vector<int> &batch_sizes) {
    std::vector<TensorListShape<>> shapes;
    for (auto batch_size : batch_sizes) {
      TensorListShape<> shape(batch_size, 2);
      for (int i = 0; i < batch_size; ++i) {
        shape.set_tensor_shape(i, TensorShape<>(i + 1, 50));
      }
      shapes.push_back(shape);
    }
    std::vector<std::vector<float>> input_buffers(batch_sizes.size());
    auto input = RandomInput(input_buffers, inp_name, shapes, [&]() { return dist(rand); });
    auto output = executor.Run({input});
    REQUIRE(cat_list_shapes(shapes) == output[0].shape);
    size_t inp_size = 0;
    for (auto &inp_buffer : input_buffers)
      inp_size += inp_buffer.size();
    std::vector<float> output_buffer(inp_size);
    std::vector<ODescr> output_vec(1);
    auto &outdesc = output_vec[0];
    OBufferDescr buf_descr;
    buf_descr.device = device_type_t::CPU;
    buf_descr.data = output_buffer.data();
    buf_descr.size = output_buffer.size() * sizeof(decltype(output_buffer)::size_type);
    outdesc.buffers = {buf_descr};
    executor.PutOutputs(output_vec);
    size_t out_i = 0;
    int i = 0;
    for (auto &inp_buffer : input_buffers) {
      for (size_t i = 0; i < inp_buffer.size(); ++i) {
        REQUIRE(output_buffer[out_i] == inp_buffer[i] * 2);
        ++out_i;
      }
    }
  };

  SECTION("Simple execute") {
    scaling_test({3, 2, 1});
    scaling_test({5});
  }

  SECTION("Repeat batch size") {
    scaling_test({3, 3});
    scaling_test({6});
  }
}

TEST_CASE("RN50 pipeline") {
  std::string pipeline_s((const char *)pipelines::rn50_gpu_dali_chr, pipelines::rn50_gpu_dali_len);
  DaliPipeline pipeline(pipeline_s, 1, 3, 0);
  DaliExecutor executor(std::move(pipeline));
  IDescr input;
  input.meta.name = "DALI_INPUT_0";
  input.meta.type = dali_data_type_t::DALI_UINT8;
  input.meta.shape = TensorListShape<1>(1);
  input.meta.shape.set_tensor_shape(0, TensorShape<>(data::jpeg_image_len));
  IBufferDescr ibuffer;
  ibuffer.data = data::jpeg_image_str;
  ibuffer.size = data::jpeg_image_len;
  ibuffer.device = device_type_t::CPU;
  input.buffers = {ibuffer};

  auto execute_with_image = [&]() {
    const float expected_values[] = {-2.1179, -2.03571, -1.80444};  // 0 values after normalization
    const int output_c = 3, output_h = 224, output_w = 224;
    auto output = executor.Run(std::vector<IDescr>({input}));
    REQUIRE(output[0].shape.tensor_shape(0) == TensorShape<3>(output_c, output_h, output_w));
    std::vector<float> output_buffer(output[0].shape.num_elements());
    std::vector<ODescr> output_vec(1);
    auto &outdesc = output_vec[0];
    OBufferDescr obuffer;
    obuffer.device = device_type_t::CPU;
    obuffer.device_id = 0;
    obuffer.data = output_buffer.data();
    obuffer.size = output_buffer.size() * sizeof(decltype(output_buffer)::size_type);
    outdesc.buffers = {obuffer};
    executor.PutOutputs(output_vec);
    for (int c = 0; c < output_c; ++c) {
      for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
          REQUIRE(output_buffer[x + (y + c * output_h) * output_w] == Approx(expected_values[c]));
        }
      }
    }
  };

  SECTION("Simple execute") {
    execute_with_image();
  }

  SECTION("Recover from error") {
    auto rand_inp_shape = TensorListShape<1>(1);
    rand_inp_shape.set_tensor_shape(0, TensorShape<>(1024));
    std::vector<std::vector<uint8_t>> rand_input_buffer(1);
    std::mt19937 rand(1217);
    std::uniform_int_distribution<short> dist(0, 255);
    auto gen = [&]() {
      return dist(rand);
    };
    auto rand_input = RandomInput(rand_input_buffer, input.meta.name, {rand_inp_shape}, gen);
    REQUIRE_THROWS(executor.Run(std::vector<IDescr>({rand_input})));

    REQUIRE_NOTHROW(execute_with_image());
  }
}

}}}}  // namespace triton::backend::dali::test
