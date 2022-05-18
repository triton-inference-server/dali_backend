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

cmake_minimum_required(VERSION 3.17)

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
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(WERROR "Trigger error on warnings" OFF)

# Don't update the TRITON_BACKEND_API_VERSION unless necessary.
# We want to use as-old-as-possible version.
set(TRITON_BACKEND_API_VERSION "r21.05" CACHE STRING "Triton backend API tag")
set(TRITON_COMMON_REPO_TAG ${TRITON_BACKEND_API_VERSION} CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG ${TRITON_BACKEND_API_VERSION} CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG ${TRITON_BACKEND_API_VERSION} CACHE STRING "Tag for triton-inference-server/backend repo")

set(DALI_DOWNLOAD_VERSION "" CACHE STRING
    "DALI version that should be downloaded by the build system.
    By default the latest available DALI version will be downloaded.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")

set(DALI_DOWNLOAD_TYPE "release" CACHE STRING
    "Type of the DALI version to download. Accepted values are: release, weekly, nightly.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD in ON")

if (${DALI_DOWNLOAD_TYPE} STREQUAL "nightly")
  set(DALI_DOWNLOAD_EXTRA_INDEX_URL "https://developer.download.nvidia.com/compute/redist/nightly" CACHE STRING
      "URL of the Index, from which DALI will be downloaded by the build system.
      Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")
elseif (${DALI_DOWNLOAD_TYPE} STREQUAL "weekly")
  set(DALI_DOWNLOAD_EXTRA_INDEX_URL "https://developer.download.nvidia.com/compute/redist/weekly" CACHE STRING
      "URL of the Index, from which DALI will be downloaded by the build system.
      Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")
else ()
  set(DALI_DOWNLOAD_EXTRA_INDEX_URL "https://developer.download.nvidia.com/compute/redist" CACHE STRING
      "URL of the Index, from which DALI will be downloaded by the build system.
      Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")
endif ()

set(DALI_DOWNLOAD_EXTRA_OPTIONS "" CACHE STRING
    "Extra options for `pip download` call, that downloads DALI wheel.
    Ignored, when TRITON_DALI_SKIP_DOWNLOAD is ON.")

if (NOT ${TRITON_DALI_SKIP_DOWNLOAD})
    message(STATUS "Building with DALI version (latest if empty): ${DALI_DOWNLOAD_VERSION}")
    message(STATUS "Downloading DALI from: ${DALI_DOWNLOAD_EXTRA_INDEX_URL}")
endif ()


# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

if (${TRITON_ENABLE_GPU})
    find_package(CUDAToolkit REQUIRED)
endif ()  # TRITON_ENABLE_GPU

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -Wno-unused-local-typedefs")

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
