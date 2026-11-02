cmake_minimum_required(VERSION 3.16)
project(dual_affine_layernorm_examples)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOC_VERSION ascend910_9382 CACHE STRING "Exact target SoC")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
if(EXISTS "$ENV{ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_DIR "$ENV{ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
else()
    set(ASCENDC_CMAKE_DIR "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
set(OPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../build/packages/vendors/dual_affine_layernorm_custom" CACHE PATH "Built operator package")

find_path(OP_INCLUDE aclnn_dual_affine_layer_norm.h PATHS "${OPP_ROOT}/op_api/include" NO_DEFAULT_PATH REQUIRED)
find_library(OP_LIBRARY cust_opapi PATHS "${OPP_ROOT}/op_api/lib" NO_DEFAULT_PATH REQUIRED)
add_executable(aclnn_dual_affine_layer_norm aclnn_dual_affine_layer_norm.cpp)
target_include_directories(aclnn_dual_affine_layer_norm PRIVATE "${OP_INCLUDE}" "$ENV{ASCEND_HOME_PATH}/include")
target_link_libraries(aclnn_dual_affine_layer_norm PRIVATE "${OP_LIBRARY}" "$ENV{ASCEND_HOME_PATH}/lib64/libascendcl.so" "$ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so")

ascendc_library(dual_affine_layer_norm_direct SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernel_direct/dual_affine_layer_norm_direct.cpp"
)
set_target_properties(dual_affine_layer_norm_direct_interface PROPERTIES
    INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
)
add_executable(direct_dual_affine_layer_norm direct_dual_affine_layer_norm.cpp)
target_include_directories(direct_dual_affine_layer_norm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../kernel_direct" "$ENV{ASCEND_HOME_PATH}/include")
target_link_libraries(direct_dual_affine_layer_norm PRIVATE dual_affine_layer_norm_direct "$ENV{ASCEND_HOME_PATH}/lib64/libascendcl.so" "$ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so")
