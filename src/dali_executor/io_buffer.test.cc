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

#include "src/dali_executor/io_buffer.h"
#include "src/dali_executor/utils/dali.h"

namespace triton { namespace backend { namespace dali { namespace test {

template<device_type_t Dev>
void test_buffer(IOBuffer<Dev> &buffer) {
  const uint8_t N = 10;
  const size_t size = N * (N + 1) / 2;
  buffer.resize(size);
  auto descriptor = buffer.get_descr();
  REQUIRE(descriptor.size == size);
  char *dst = reinterpret_cast<char *>(descriptor.data);
  for (uint8_t i = 1; i <= N; ++i) {
    std::vector<uint8_t> chunk(i, i);
    MemCopy(Dev, dst, device_type_t::CPU, chunk.data(), i);
    dst += i;
  }
  // validation
  std::vector<uint8_t> result(size);
  REQUIRE(descriptor.device == Dev);
  MemCopy(device_type_t::CPU, result.data(), Dev, descriptor.data, size);
  size_t it = 0;
  for (uint8_t i = 1; i <= N; ++i) {
    for (uint8_t j = 0; j < i; ++j) {
      REQUIRE(result[it] == i);
      ++it;
    }
  }
}

TEST_CASE("IOBuffer<CPU> extend & copy") {
  IOBuffer<device_type_t::CPU> buffer;

  SECTION("Copy") {
    test_buffer(buffer);
  }
}

TEST_CASE("IOBuffer<GPU> extend & copy") {
  IOBuffer<device_type_t::GPU> buffer;

  SECTION("Copy") {
    test_buffer(buffer);
  }
}


}}}}  // namespace triton::backend::dali::test
