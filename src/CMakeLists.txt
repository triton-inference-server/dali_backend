# The MIT License (MIT)
#
# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

configure_file(libtriton_dali.ldscript libtriton_dali.ldscript COPYONLY)

add_subdirectory(dali_executor)

add_library(
        triton-dali-backend-utils STATIC
        utils/triton.cc
        config_tools/config_tools.cc
)

target_include_directories(
        triton-dali-backend-utils PUBLIC
        ${tritondalibackend_SOURCE_DIR}
        ${DALI_INCLUDE_DIR}
)

target_link_libraries(
        triton-dali-backend-utils PUBLIC
        triton-core-serverstub
        triton-backend-utils
        dali_executor
)

set_target_properties(triton-dali-backend-utils PROPERTIES CXX_STANDARD 17)


add_library(
        triton-dali-backend SHARED
        dali_backend.cc
        dali_model_instance.cc
        dali_model.cc
)

add_library(
        TritonDaliBackend::triton-dali-backend ALIAS triton-dali-backend
)


target_include_directories(
        triton-dali-backend
        PRIVATE
        ${tritondalibackend_SOURCE_DIR}
)

target_compile_options(
        triton-dali-backend PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra -Wno-unused-parameter -Werror -Wno-unused-function -Wno-type-limits>
)

target_link_libraries(
        triton-dali-backend
        PRIVATE
        dali_executor
        triton-dali-backend-utils
)

set_target_properties(
        triton-dali-backend
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME triton_dali
            LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_dali.ldscript
            LINK_FLAGS "-Wl,--version-script libtriton_dali.ldscript"
            SKIP_BUILD_RPATH TRUE
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH FALSE
            INSTALL_RPATH "$\{ORIGIN\}/dali"
            CXX_STANDARD 17
)

set(
        DALI_BACKEND_TEST_SRCS
        dali_executor/main.test.cc
        dali_executor/executor.test.cc
        dali_executor/io_buffer.test.cc
        utils/utils.test.cc
        config_tools/config_tools.test.cc
)

add_executable(unittests ${DALI_BACKEND_TEST_SRCS})
target_link_libraries(
        unittests
        PRIVATE
        Catch2::Catch2
        dali_executor
        triton-dali-backend-utils
        ${CMAKE_DL_LIBS}
)

set_target_properties(unittests PROPERTIES CXX_STANDARD 17)
