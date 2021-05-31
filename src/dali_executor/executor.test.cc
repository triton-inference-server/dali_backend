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

template<typename T, typename Op>
void coalesced_compare(const std::vector<OBufferDescr> &obuffers,
                       const std::vector<std::vector<T>> &ibuffers, size_t inp_size, const Op &op) {
  size_t inp_buff_i = 0;
  size_t inp_i = 0;
  size_t out_buff_i = 0;
  size_t out_i = 0;
  std::vector<T> obuffer;
  for (size_t i = 0; i < inp_size; ++i) {
    if (inp_i == ibuffers[inp_buff_i].size()) {
      inp_i = 0;
      inp_buff_i++;
    }
    if (out_i == obuffers[out_buff_i].size / sizeof(T)) {
      out_i = 0;
      out_buff_i++;
    }
    if (out_i == 0) {
      auto descr = obuffers[out_buff_i];
      REQUIRE(descr.size % sizeof(T) == 0);
      obuffer.resize(descr.size / sizeof(T));
      MemCopy(CPU, obuffer.data(), descr.device, descr.data, descr.size);
    }
    REQUIRE(obuffer[out_i] == op(ibuffers[inp_buff_i][inp_i]));
    out_i++;
    inp_i++;
  }
}

TEST_CASE("Scaling Pipeline") {
  std::string pipeline_s((const char *)pipelines::scale_pipeline_str,
                         pipelines::scale_pipeline_len);
  DaliPipeline pipeline(pipeline_s, 256, 4, 0);
  DaliExecutor executor(std::move(pipeline));
  std::mt19937 rand(1217);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const std::string inp_name = "INPUT0";
  auto scaling_test = [&](const std::vector<int> &batch_sizes,
                          const std::vector<int> &out_batch_sizes,
                          const std::vector<device_type_t> &out_devs) {
    REQUIRE(std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0) ==
            std::accumulate(out_batch_sizes.begin(), out_batch_sizes.end(), 0));
    REQUIRE(out_devs.size() == out_batch_sizes.size());
    std::vector<TensorListShape<>> shapes;
    for (auto batch_size : batch_sizes) {
      TensorListShape<> shape(batch_size, 2);
      for (int i = 0; i < batch_size; ++i) {
        shape.set_tensor_shape(i, TensorShape<>(i + 1, 50));
      }
      shapes.push_back(shape);
    }
    std::vector<std::vector<float>> input_buffers;
    auto input = RandomInput(input_buffers, inp_name, shapes, [&]() { return dist(rand); });
    auto output = executor.Run({input});
    REQUIRE(cat_list_shapes(shapes) == output[0].shape);
    size_t inp_size = 0;
    for (auto &inp_buffer : input_buffers)
      inp_size += inp_buffer.size();
    std::vector<std::unique_ptr<IOBufferI>> output_buffers;
    int ti = 0;
    for (size_t out_i = 0; out_i < out_batch_sizes.size(); ++out_i) {
      int64_t buffer_vol = 0;
      for (int i = 0; i < out_batch_sizes[out_i]; ++i) {
        buffer_vol += volume(output[0].shape[ti]) * sizeof(float);
        ti++;
      }
      if (out_devs[out_i] == device_type_t::CPU) {
        output_buffers.emplace_back(std::make_unique<IOBuffer<CPU>>(buffer_vol));
      } else {
        output_buffers.emplace_back(std::make_unique<IOBuffer<GPU>>(buffer_vol));
      }
    }
    std::vector<ODescr> output_vec(1);
    auto &outdesc = output_vec[0];
    for (auto &out_buffer : output_buffers) {
      outdesc.buffers.push_back(out_buffer->get_descr());
    }
    executor.PutOutputs(output_vec);
    coalesced_compare(outdesc.buffers, input_buffers, inp_size, [](float a) { return a * 2; });
  };

  SECTION("Simple execute") {
    scaling_test({3, 2, 1}, {6}, {CPU});
    scaling_test({5}, {5}, {GPU});
  }

  SECTION("Chunked output") {
    scaling_test({3, 3}, {3, 3}, {CPU, CPU});
    scaling_test({6}, {2, 4}, {GPU, GPU});
    scaling_test({8}, {6, 2}, {CPU, GPU});
    scaling_test({64}, {32, 16, 16}, {CPU, GPU, GPU});
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
    obuffer.size = output_buffer.size() * sizeof(decltype(output_buffer)::value_type);
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
    std::vector<std::vector<uint8_t>> rand_input_buffer;
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
