cmake_minimum_required(VERSION 3.16.0)
project(dual_affine_layernorm_op_prj)
find_package(ASC REQUIRED)
set(CANNJUDGE_COMPILE_JOBS 4 CACHE STRING "Parallel operator compile jobs")
set(CANN_HOST_ARCH "${CMAKE_SYSTEM_PROCESSOR}-linux")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910_93)
endif()
set(package_name dual_affine_layernorm_custom)
npu_op_package(${package_name}
    TYPE RUN
    CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}
)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
