cmake_minimum_required(VERSION 3.25)
project(comet_super_build LANGUAGES CXX C)
include(ExternalProject)

# Adopted from https://gist.github.com/scivision/bb1d47a9529e153617414e91ff5390af
find_package(Git REQUIRED)

function(init_git_submodule dir)
# Init a Git submodule 
# 
# include(AddGitSubmodule.cmake)
# add_git_submodule(mysubmod_dir)

if(NOT EXISTS ${dir}/README.md)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --depth 1 -- ${dir}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
endif()

endfunction(init_git_submodule)
#-------------------------------------------------------------------------------

message(STATUS "Running on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Architectures to build for on macOS")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
  set(DETECTED_CPU_ARCH "AArch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  set(DETECTED_CPU_ARCH "X86")
else()
  message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

init_git_submodule(${CMAKE_SOURCE_DIR}/blis)
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
set(LLVM_TARGETS "${DETECTED_CPU_ARCH}" CACHE STRING "LLVM targets to build")
option(ENABLE_AMD_GPU_BACKEND OFF) 
option(ENABLE_NVIDIA_GPU_BACKEND OFF) 
option(ENABLE_FPGA_TARGET OFF)
if(ENABLE_AMD_GPU_BACKEND)
    set(LLVM_TARGETS "${LLVM_TARGETS},AMDGPU")
endif()

if(ENABLE_NVIDIA_GPU_BACKEND)
    set(LLVM_TARGETS "${LLVM_TARGETS},NVPTX")
endif()
set(DEPENDENCIES llvm blis)
if(NOT LLVM_CUSTOM_BUILD_PATH)
    init_git_submodule(${CMAKE_SOURCE_DIR}/llvm)
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
    init_git_submodule(${CMAKE_SOURCE_DIR}/triton)
    set(DEVICE_COMPUTE_CAPABILITY "" CACHE STRING "Device compute capability for Triton codegen")
    ExternalProject_Add(
        triton
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/triton
        BINARY_DIR ${CMAKE_BINARY_DIR}/triton
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/triton.patch
    )
    set(TRITON_PATH ${CMAKE_SOURCE_DIR}/triton)
    set(TRITON_BUILD_PATH ${CMAKE_BINARY_DIR}/triton)
    list(APPEND DEPENDENCIES llvm-spirv mcl triton)
endif()

if(ENABLE_FPGA_TARGET)
    init_git_submodule(${CMAKE_SOURCE_DIR}/tools/llvm-spirv)
    init_git_submodule(${CMAKE_SOURCE_DIR}/tools/spirv-llvm-translate)
    file(CREATE_LINK ${CMAKE_SOURCE_DIR}/tools/spirv-llvm-translate ${CMAKE_SOURCE_DIR}/tools/llvm-spirv/llvm/projects/spirv-llvm-translate SYMBOLIC)
    # ADD_CUSTOM_TARGET(link_spirv_translate ALL
    #               COMMAND ${CMAKE_COMMAND} -E create_symlink ${target} ${link})
    ExternalProject_Add(
        llvm-spirv
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/tools/llvm-spirv
        BINARY_DIR ${CMAKE_BINARY_DIR}/tools/llvm-spirv
        CONFIGURE_COMMAND cmake -G Ninja ${CMAKE_SOURCE_DIR}/tools/llvm-spirv/llvm/ -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=host
        BUILD_COMMAND ninja llvm-spirv && ninja llvm-dis
        INSTALL_COMMAND ""
    )

    init_git_submodule(${CMAKE_SOURCE_DIR}/runtimes/mcl)
    ExternalProject_Add(
        mcl
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/runtimes/mcl
        BINARY_DIR ${CMAKE_BINARY_DIR}/runtimes/mcl
        CONFIGURE_COMMAND cd ${CMAKE_SOURCE_DIR}/runtimes/mcl && autoreconf --install && ./configure --prefix=${CMAKE_BINARY_DIR}/runtimes/mcl 
        BUILD_COMMAND cd ${CMAKE_SOURCE_DIR}/runtimes/mcl && make -j && make install
        INSTALL_COMMAND ""
    )
    set(MCL_BUILD_PATH ${CMAKE_BINARY_DIR}/runtimes/mcl)
    list(APPEND DEPENDENCIES llvm-spirv mcl)
endif()


ExternalProject_Add(
    comet
    BINARY_DIR ${CMAKE_BINARY_DIR}/comet
    LIST_SEPARATOR ","
    PREFIX ${CMAKE_BINARY_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_COMMAND cmake -G Ninja ${CMAKE_CURRENT_SOURCE_DIR}/cmake -DMLIR_DIR=${LLVM_BUILD_PATH}/lib/cmake/mlir -DLLVM_DIR=${LLVM_BUILD_PATH}/lib/cmake/llvm -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/blis/ -DBLAS_INCLUDE_DIRS=${CMAKE_BINARY_DIR}/blis/include/blis/ -DENABLE_AMD_GPU_BACKEND=${ENABLE_AMD_GPU_BACKEND} -DENABLE_NVIDIA_GPU_BACKEND=${ENABLE_NVIDIA_GPU_BACKEND} -DENABLE_FPGA_TARGET=${ENABLE_FPGA_TARGET} -DDEVICE_COMPUTE_CAPABILITY=${DEVICE_COMPUTE_CAPABILITY} -DTRITON_PATH=${TRITON_PATH} -DTRITON_BUILD_PATH=${TRITON_BUILD_PATH} -DMCL_BUILD_PATH=${MCL_BUILD_PATH} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    BUILD_COMMAND cmake --build . -- -j 8
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ""
    DEPENDS ${DEPENDENCIES}
)