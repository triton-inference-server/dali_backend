// The MIT License (MIT)
//
// Copyright (c) 2021 NVIDIA CORPORATION
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

#include "src/dali_executor/io_descriptor.h"

namespace triton { namespace backend { namespace dali { namespace test {


TEST_CASE("IODescriptor owning memory")
{
  std::mt19937 g;
  std::uniform_int_distribution<int> buffer_values(0, 255);
  std::uniform_int_distribution<int> buffer_sizes(1, 1e6);
  std::vector<std::vector<char>> buffers;
  auto n_buffers = 10;
  for (int i = 0; i < n_buffers; i++) {
    std::vector<char> buffer;
    std::generate_n(std::back_inserter(buffer), buffer_sizes(g), [&]() {
      return buffer_values(g);
    });
    buffers.push_back(buffer);
  }
  int stitched_buffer_size = 0;
  for (const auto& buf : buffers) {
    stitched_buffer_size += buf.size();
  }
  IODescr<true> desc(stitched_buffer_size);
  REQUIRE(desc.capacity() == stitched_buffer_size);

  for (const auto& buf : buffers) {
    desc.append(make_span(buf));
  }

  auto* ptr = desc.buffer.data();
  REQUIRE(desc.buffer.size() == stitched_buffer_size);
  for (int i = 0; i < buffers.size(); i++) {
    auto& buf = buffers[i];
    for (int j = 0; j < buf.size(); j++) {
      REQUIRE(buf[j] == *ptr++);
    }
  }
}


}}}}  // namespace triton::backend::dali::test