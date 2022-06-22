# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION
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

set(
     DALI_BACKEND_SRCS
        dali_executor.cc
        dali_pipeline.cc
        io_buffer.cc
)

include(${tritondalibackend_SOURCE_DIR}/cmake/dali.cmake)

if(DALI_DOWNLOAD_EXTRA_OPTIONS)
    configure_file(${tritondalibackend_SOURCE_DIR}/cmake/dalienv-extra_args.yml.in dalienv.yml @ONLY)
else()
    configure_file(${tritondalibackend_SOURCE_DIR}/cmake/dalienv.yml.in dalienv.yml @ONLY)
endif()

add_custom_command(
    OUTPUT dali_env
    COMMAND
        conda env create -f dalienv.yml -p ${CMAKE_CURRENT_BINARY_DIR}/dalienv -q
    COMMAND touch dali_env  # So that this command won't be re-run every time
    COMMENT "Building DALI conda environment."
)

if(${TRITON_DALI_SKIP_DOWNLOAD})
    get_dali_paths(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
else()
    set(DALI_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dalienv/lib/python3.8/site-packages/nvidia/dali/include)
    set(DALI_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/dalienv/lib/python3.8/site-packages/nvidia/dali)
    set(DALI_LIBRARIES dali dali_core dali_kernels dali_operators)
endif()  # TRITON_DALI_SKIP_DOWNLOAD

message(STATUS "DALI includes dir: " ${DALI_INCLUDE_DIR})
message(STATUS "DALI libs dir: " ${DALI_LIB_DIR})
message(STATUS "DALI libs: " ${DALI_LIBRARIES})

add_library(
        dali_executor
        STATIC
        ${DALI_BACKEND_SRCS}
        $<$<NOT:$<BOOL:${TRITON_DALI_SKIP_DOWNLOAD}>>:dali_env>
)

target_include_directories(dali_executor PUBLIC BEFORE
        ${tritondalibackend_SOURCE_DIR}
        ${DALI_INCLUDE_DIR})

target_link_directories(dali_executor PUBLIC
        ${DALI_LIB_DIR})

target_link_libraries(dali_executor PUBLIC
        CUDA::cudart
        ${DALI_LIBRARIES})

install(
        DIRECTORY
        ${DALI_LIB_DIR}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/dali
)
