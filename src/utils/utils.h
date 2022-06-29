// The MIT License (MIT)
//
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
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

#ifndef DALI_BACKEND_UTILS_UTILS_H_
#define DALI_BACKEND_UTILS_UTILS_H_

#include <ctime>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

namespace triton { namespace backend { namespace dali {

/**
 * Splits string according to a delimiter.
 *
 * This function doesn't include empty strings in the output.
 *
 * @param str String to be split
 * @param delimiter Delimiter. Won't be included in the output
 * @return
 */
inline std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
  std::vector<std::string> ret;
  for (size_t start = 0, len = str.length(); start < len;) {
    auto end = str.find(delimiter, start);
    if (end == std::string::npos) {
      end = len;
    }
    if (end > start) {
      ret.emplace_back(str, start, end - start);
    }
    start = end + delimiter.length();
  }
  return ret;
}


template<typename T>
T from_string(const std::string& str);

template<>
inline int from_string<int>(const std::string& str) {
  return std::stoi(str);
}

template<>
inline std::string from_string<std::string>(const std::string& str) {
  return str;
}

template <typename T>
std::string vec_to_string(const std::vector<T> &vec, const std::string &lbracket = "{",
                          const std::string &rbracket = "}", const std::string &delim = ", ") {
  std::stringstream ss;
  ss << lbracket;
  auto it = vec.begin();
  if (vec.size() > 0) {
    ss << *it;
    for (++it; it != vec.end(); ++it) {
      ss << delim;
      ss << *it;
    }
  }
  ss << rbracket;
  return ss.str();
}


inline std::string timestamp() {
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::stringstream timestamp;
  timestamp << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return timestamp.str();
}

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_UTILS_UTILS_H_
