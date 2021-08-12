// The MIT License (MIT)
//
// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
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

#ifndef TRITONDALIBACKEND_BACKTRACE_H
#define TRITONDALIBACKEND_BACKTRACE_H

#ifdef IS_DEBUG_CONFIG
#include "boost/stacktrace.hpp"
#endif

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace dali {

inline void print_backtrace() {
#ifdef IS_DEBUG_CONFIG
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              make_string("Backtrace:\n", boost::stacktrace::stacktrace()).c_str());
#endif
}

}}}  // namespace triton::backend::dali


#endif  // TRITONDALIBACKEND_BACKTRACE_H
