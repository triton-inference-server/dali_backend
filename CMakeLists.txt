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

cmake_minimum_required(VERSION 3.18)

project(tritondalibackend C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE})

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
option(TRITON_DALI_SKIP_DOWNLOAD "Don't download DALI. Use the version provided in the system" OFF)
option(TRITON_DALI_BUILD_IN_TRITON "Turn this option on, if the DALI Backend is built inside Triton docker container." OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(WERROR "Trigger error on warnings" OFF)

# Don't update the TRITON_BACKEND_API_VERSION unless necessary.
# We want to use as-old-as-possible version.
set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_BACKEND_API_VERSION "r21.05" CACHE STRING "Triton backend API tag")
set(TRITON_COMMON_REPO_TAG ${TRITON_BACKEND_API_VERSION} CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG ${TRITON_BACKEND_API_VERSION} CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG ${TRITON_BACKEND_API_VERSION} CACHE STRING "Tag for triton-inference-server/backend repo")

set(PYTHON_VERSION "3.10" CACHE STRING
    "Python version, which will be used to create conda environment for DALI.")

set(DALI_VERSION "" CACHE STRING
    "DALI version that should be downloaded by the build system.
    By default the latest available DALI version will be downloaded.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")

set(DALI_EXTRA_INDEX_URL "https://developer.download.nvidia.com/compute/redist" CACHE STRING
    "URL of the Index, from which DALI will be downloaded by the build system.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")

set(DALI_DOWNLOAD_EXTRA_OPTIONS "" CACHE STRING
    "Extra options for `pip install` call, that downloads DALI wheel.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")

# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

if (${TRITON_ENABLE_GPU})
    find_package(CUDAToolkit REQUIRED)
endif ()  # TRITON_ENABLE_GPU

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -Wno-unused-local-typedefs -Wno-free-nonheap-object")

set(DALI_DOWNLOAD_PKG_NAME "nvidia-dali-cuda${CUDAToolkit_VERSION_MAJOR}0" CACHE STRING
        "DALI pip package name to download.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")

if(NOT DALI_VERSION AND NOT DALI_DOWNLOAD_PKG_NAME MATCHES "nightly" AND NOT TRITON_DALI_SKIP_DOWNLOAD)
    file(READ ${tritondalibackend_SOURCE_DIR}/DALI_VERSION DALI_VERSION)
    string(STRIP "${DALI_VERSION}" DALI_VERSION)
endif()

if (${TRITON_DALI_SKIP_DOWNLOAD})
    message(STATUS "Skipping DALI download.")
else ()
    message(STATUS "Building with DALI version: ${DALI_VERSION}")
    message(STATUS "DALI wheel extra index url: ${DALI_EXTRA_INDEX_URL}")
    message(STATUS "DALI wheel package name: ${DALI_DOWNLOAD_PKG_NAME}")
    message(STATUS "Downloading DALI with extra options: ${DALI_DOWNLOAD_EXTRA_OPTIONS}")
endif ()

if (WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

add_subdirectory(src)

add_subdirectory(extern/Catch2)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonDaliBackend)

install(
  TARGETS
    triton-dali-backend
  EXPORT
    triton-dali-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/dali
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/dali
)


install(
  EXPORT
    triton-dali-backend-targets
  FILE
    TritonDaliBackendTargets.cmake
  NAMESPACE
    TritonDaliBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonDaliBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonDaliBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonDaliBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-dali-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonDaliBackendTargets.cmake
  NAMESPACE TritonDaliBackend::
)

export(PACKAGE TritonDaliBackend)
