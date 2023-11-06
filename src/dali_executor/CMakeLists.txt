# The MIT License (MIT)
#
# Copyright (c) 2020-2022 NVIDIA CORPORATION
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
    set(CONF_DOWNLOAD_EXTRA_OPTIONS "- ${DALI_DOWNLOAD_EXTRA_OPTIONS}")
endif()

set(CONF_DOWNLOAD_PKG "${DALI_DOWNLOAD_PKG_NAME}")
if(DALI_VERSION)
    string(APPEND CONF_DOWNLOAD_PKG "==${DALI_VERSION}")
endif()

configure_file(${tritondalibackend_SOURCE_DIR}/cmake/dalienv.yml.in dalienv.yml @ONLY)

set(CONDA_ENV_PREFIX ${CMAKE_INSTALL_PREFIX}/backends/dali/conda/envs)
set(CONDA_ENV_PATH ${CONDA_ENV_PREFIX}/dalienv)

add_custom_command(
    OUTPUT dali_env
    COMMAND
        conda env create -f dalienv.yml -p ${CMAKE_CURRENT_BINARY_DIR}/dalienv --force -q && conda update -p ${CMAKE_CURRENT_BINARY_DIR}/dalienv --all -c conda-forge -y -q && curl https://patch-diff.githubusercontent.com/raw/python/cpython/pull/99421.patch | patch -d ${CMAKE_CURRENT_BINARY_DIR}/dalienv/lib/python3.10/ -p2 -s -t || true
    COMMAND touch dali_env  # So that this command won't be re-run every time
    COMMENT "Building DALI conda environment."
)

if(${TRITON_DALI_SKIP_DOWNLOAD})
    get_dali_paths(${CONDA_ENV_PATH} DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
else()
    set(DALI_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dalienv/lib/python${PYTHON_VERSION}/site-packages/nvidia/dali/include)
    set(DALI_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/dalienv/lib/python${PYTHON_VERSION}/site-packages/nvidia/dali)
    set(DALI_LIBRARIES dali dali_core dali_kernels dali_operators)
endif()  # TRITON_DALI_SKIP_DOWNLOAD

set(DALI_INCLUDE_DIR ${DALI_INCLUDE_DIR} PARENT_SCOPE)
set(DALI_LIB_DIR ${DALI_LIB_DIR} PARENT_SCOPE)
set(DALI_LIBRARIES ${DALI_LIBRARIES} PARENT_SCOPE)

message(STATUS "DALI includes dir: " ${DALI_INCLUDE_DIR})
message(STATUS "DALI libs dir: " ${DALI_LIB_DIR})
message(STATUS "DALI libs: " ${DALI_LIBRARIES})

add_library(
        dali_executor
        STATIC
        ${DALI_BACKEND_SRCS}
        $<$<NOT:$<BOOL:${TRITON_DALI_SKIP_DOWNLOAD}>>:dali_env>
)

set_property(TARGET dali_executor PROPERTY CXX_STANDARD 17)
target_include_directories(dali_executor PUBLIC BEFORE
        ${tritondalibackend_SOURCE_DIR}
        ${DALI_INCLUDE_DIR})

target_link_directories(dali_executor PUBLIC
        ${DALI_LIB_DIR})

target_link_libraries(dali_executor PUBLIC
        CUDA::cudart
        ${DALI_LIBRARIES})

target_compile_definitions(dali_executor PUBLIC)
if (NOT ${TRITON_DALI_SKIP_DOWNLOAD})
    install(DIRECTORY ${CONDA_ENV_PATH}
            DESTINATION ${CONDA_ENV_PREFIX}
            USE_SOURCE_PERMISSIONS)
endif ()
