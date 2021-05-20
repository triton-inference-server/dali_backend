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

# Using DALI's installed whl, find the location of DALI libs and DALI include dirs
function(get_dali_paths DALI_INCLUDE_DIR_VAR DALI_LIB_DIR_VAR DALI_LIBRARIES_VAR)

    execute_process(
            COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_include_dir())"
            OUTPUT_VARIABLE DALI_INCLUDE_DIR
            RESULT_VARIABLE INCLUDE_DIR_RESULT)
    string(STRIP ${DALI_INCLUDE_DIR} DALI_INCLUDE_DIR)

    if (${INCLUDE_DIR_RESULT} EQUAL "1")
        message(FATAL_ERROR "Failed to get include paths for DALI. Make sure that DALI is installed.")
    endif ()

    execute_process(
            COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
            OUTPUT_VARIABLE DALI_LIB_DIR
            RESULT_VARIABLE LIB_DIR_RESULT)
    string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

    if (${LIB_DIR_RESULT} EQUAL "1")
        message(FATAL_ERROR "Failed to get library paths for DALI. Make sure that DALI is installed.")
    endif ()

    set(${DALI_INCLUDE_DIR_VAR} ${DALI_INCLUDE_DIR} PARENT_SCOPE)
    set(${DALI_LIB_DIR_VAR} ${DALI_LIB_DIR} PARENT_SCOPE)
    set(${DALI_LIBRARIES_VAR} dali dali_core dali_kernels dali_operators PARENT_SCOPE)
endfunction()
