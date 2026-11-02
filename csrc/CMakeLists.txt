cmake_minimum_required(VERSION 3.16)
project(dual_affine_layernorm_torch LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYTHON_EXECUTABLE "" CACHE FILEPATH "Target Python interpreter")
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE is required")
endif()
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__file__, end='')" OUTPUT_VARIABLE TORCH_FILE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch_npu; print(torch_npu.__file__, end='')" OUTPUT_VARIABLE TORCH_NPU_FILE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['include'], end='')" OUTPUT_VARIABLE PYTHON_INCLUDE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')" OUTPUT_VARIABLE TORCH_CXX11_ABI COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(TORCH_ROOT "${TORCH_FILE}" DIRECTORY)
get_filename_component(TORCH_NPU_ROOT "${TORCH_NPU_FILE}" DIRECTORY)
set(SOC_VERSION ascend910_9382 CACHE STRING "Exact target SoC")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
if(EXISTS "$ENV{ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_DIR "$ENV{ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
else()
    set(ASCENDC_CMAKE_DIR "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
set(CMAKE_SKIP_RPATH FALSE)
ascendc_library(dual_affine_layer_norm_torch_kernel SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernel_direct/dual_affine_layer_norm_direct.cpp"
)
set_target_properties(dual_affine_layer_norm_torch_kernel_interface PROPERTIES
    INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
)
set(OPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../build/packages/vendors/dual_affine_layernorm_custom" CACHE PATH "Installed private operator package")
set(TORCH_EXTENSION_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dual_affine_layernorm" CACHE PATH "Python module output directory")
set(TORCH_EXTENSION_OUTPUT_NAME "_dual_affine_layer_norm.so" CACHE STRING "Setuptools extension filename")
find_path(OP_INCLUDE aclnn_dual_affine_layer_norm.h PATHS "${OPP_ROOT}/op_api/include" NO_DEFAULT_PATH REQUIRED)
find_library(OP_LIBRARY cust_opapi PATHS "${OPP_ROOT}/op_api/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_LIBRARY torch PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(C10_LIBRARY c10 PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${TORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(C10_NPU_LIBRARY c10_npu PATHS "${TORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH)
add_library(_dual_affine_layer_norm MODULE dual_affine_layer_norm_torch.cpp)
target_compile_definitions(_dual_affine_layer_norm PRIVATE TORCH_EXTENSION_NAME=_dual_affine_layer_norm _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_include_directories(_dual_affine_layer_norm PRIVATE
    "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include"
    "${TORCH_NPU_ROOT}/include" "${PYTHON_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernel_direct" "${OP_INCLUDE}"
    "$ENV{ASCEND_HOME_PATH}/include"
)
target_link_directories(_dual_affine_layer_norm PRIVATE "${TORCH_ROOT}/lib" "${TORCH_NPU_ROOT}/lib")
target_link_libraries(_dual_affine_layer_norm PRIVATE
    dual_affine_layer_norm_torch_kernel "${OP_LIBRARY}"
    "${TORCH_LIBRARY}" "${TORCH_CPU_LIBRARY}" "${TORCH_PYTHON_LIBRARY}" "${C10_LIBRARY}"
    "${TORCH_NPU_LIBRARY}"
    "$ENV{ASCEND_HOME_PATH}/lib64/libascendcl.so" "$ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so"
    ${CMAKE_DL_LIBS}
)
if(C10_NPU_LIBRARY)
    target_link_libraries(_dual_affine_layer_norm PRIVATE "${C10_NPU_LIBRARY}")
endif()
set_target_properties(_dual_affine_layer_norm PROPERTIES
    PREFIX "" OUTPUT_NAME "${TORCH_EXTENSION_OUTPUT_NAME}" SUFFIX "" LIBRARY_OUTPUT_DIRECTORY "${TORCH_EXTENSION_OUTPUT_DIR}"
    BUILD_RPATH "\$ORIGIN;${OPP_ROOT}/op_api/lib;${TORCH_ROOT}/lib;${TORCH_NPU_ROOT}/lib"
    INSTALL_RPATH "\$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE
)
