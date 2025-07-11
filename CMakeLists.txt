cmake_minimum_required(VERSION 3.25)
project(comet_super_build LANGUAGES CXX C)

include(ExternalProject)

ExternalProject_Add(
    blis
    PREFIX ${CMAKE_BINARY_DIR}/blis
    BINARY_DIR ${CMAKE_BINARY_DIR}/blis
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/blis
    CONFIGURE_COMMAND cd ${CMAKE_SOURCE_DIR}/blis && ./configure --prefix=${CMAKE_BINARY_DIR}/blis  --disable-shared auto
    PATCH_COMMAND cd ${CMAKE_SOURCE_DIR} && patch -s  -p0 < ${CMAKE_SOURCE_DIR}/comet-blis.patch
    BUILD_COMMAND cd ${CMAKE_SOURCE_DIR}/blis && make -j && make install
    INSTALL_COMMAND ""
)

set(LLVM_CUSTOM_BUILD_PATH "" CACHE PATH "Path to existing LLVM build directory")
set(LLVM_PROJECTS "mlir,openmp,clang" CACHE STRING "LLVM projects to build")
set(LLVM_TARGETS "X86" CACHE STRING "LLVM targets to build")
option(ENABLE_AMD_GPU_BACKEND OFF) 
option(ENABLE_NVIDIA_GPU_BACKEND OFF) 
option(ENABLE_FPGA_TARGET OFF)
if(ENABLE_AMD_GPU_BACKEND)
    set(LLVM_TARGETS ${LLVM_TARGETS}, "AMDGPU")
endif()

if(ENABLE_NVIDIA_GPU_BACKEND)
    set(LLVM_TARGETS ${LLVM_TARGETS}, "NVPTX")
endif()

if(NOT LLVM_CUSTOM_BUILD_PATH)
    message(STATUS "Building LLVM from source with projects: ${LLVM_PROJECTS} and targets: ${LLVM_TARGETS}")
    ExternalProject_Add(
        llvm
        LIST_SEPARATOR ","
        PREFIX ${CMAKE_BINARY_DIR}/llvm
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/
        CONFIGURE_COMMAND cmake -G Ninja ${CMAKE_CURRENT_SOURCE_DIR}/llvm/llvm -DLLVM_ENABLE_PROJECTS=${LLVM_PROJECTS} -DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS} -DCMAKE_OSX_ARCHITECTURES=arm64 -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release
        BUILD_COMMAND cmake --build . -- -j 8
        INSTALL_COMMAND ""
    )
    set(LLVM_BUILD_PATH ${CMAKE_BINARY_DIR}/llvm/src/llvm-build)
else()
    message(STATUS "Using existing LLVM build at: ${LLVM_CUSTOM_BUILD_PATH}")
    ExternalProject_Add(
        llvm
        BINARY_DIR ${LLVM_CUSTOM_BUILD_PATH}
        SOURCE_DIR ${LLVM_CUSTOM_BUILD_PATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    
    set(LLVM_BUILD_PATH ${LLVM_CUSTOM_BUILD_PATH})
endif()

if(ENABLE_AMD_GPU_BACKEND OR ENABLE_NVIDIA_GPU_BACKEND)
ExternalProject_Add(
    triton
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/triton
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND git submodule update --init ${CMAKE_SOURCE_DIR}/triton 
    PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/triton.patch
)
endif()

ExternalProject_Add(
    comet
    BINARY_DIR ${CMAKE_BINARY_DIR}/comet
    LIST_SEPARATOR ","
    PREFIX ${CMAKE_BINARY_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_COMMAND cmake -G Ninja ${CMAKE_CURRENT_SOURCE_DIR}/cmake -DMLIR_DIR=${LLVM_BUILD_PATH}/lib/cmake/mlir -DLLVM_DIR=${LLVM_BUILD_PATH}/lib/cmake/llvm -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/blis/ -DBLAS_INCLUDE_DIRS=${CMAKE_BINARY_DIR}/blis/include/blis/ -DENABLE_AMD_GPU_BACKEND=${ENABLE_AMD_GPU_BACKEND} -DENABLE_NVIDIA_GPU_BACKEND=${ENABLE_NVIDIA_GPU_BACKEND} -DENABLE_FPGA_TARGET=${ENABLE_FPGA_TARGET} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    BUILD_COMMAND cmake --build . -- -j 8
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ""
    DEPENDS llvm blis
)