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
#include "src/dali_executor/serialized_pipelines.h"
#include "src/dali_executor/test/test_utils.h"

namespace triton { namespace backend { namespace dali { namespace test {

TEST_CASE("Distribute batch size")
{
  REQUIRE(distribute_batch_size(4) == std::vector<int>{4});
  REQUIRE(distribute_batch_size(5) == std::vector<int>{1, 4});
  REQUIRE(distribute_batch_size(1) == std::vector<int>{1});
  REQUIRE(distribute_batch_size(15) == std::vector<int>{1, 2, 4, 8});
}

TEST_CASE("Scaling Pipeline")
{
  std::string pipeline(
      (const char*)pipelines::scale_pipeline_str,
      pipelines::scale_pipeline_len);
  DaliExecutor executor(pipeline, 0);
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
    auto output = executor.Run({input});
    REQUIRE(shape == output[0].shape);
    std::vector<float> output_buffer(input_buffer.size());
    OutputDescriptor outdesc;
    outdesc.device = device_type_t::CPU;
    outdesc.buffer = make_span(
        reinterpret_cast<char*>(output_buffer.data()),
        output_buffer.size() * sizeof(decltype(output_buffer)::size_type));
    executor.PutOutputs({outdesc});
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
    auto created_pipelines = executor.NumCreatedPipelines();
    scaling_test(3);
    REQUIRE(created_pipelines == executor.NumCreatedPipelines());
  }
}

}}}}  // namespace triton::backend::dali::test
