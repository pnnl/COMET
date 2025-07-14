##===- CMakeLists.txt - COMET cmake root ----------------------*- cmake -*-===//
##
## Configure the COMET build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.25)

# If we are not building as a part of LLVM, build COMET as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------
project(comet LANGUAGES CXX C)
  
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules) 

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
option(ENABLE_AMD_GPU_BACKEND OFF) 
option(ENABLE_NVIDIA_GPU_BACKEND OFF) 
option(ENABLE_FPGA_TARGET OFF) 

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(CODEGEN_BACKENDS "" CACHE INTERNAL "TRITON BACKENDS")
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GR-")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
endif ()

if(${ENABLE_AMD_GPU_BACKEND})
  add_definitions(-DENABLE_AMD_GPU_BACKEND)
  set(CODEGEN_BACKENDS ${CODEGEN_BACKENDS}amd;nvidia) # Triton is looking for some NVGPUIR library 
                                                      # even if we only enable AMDGPU
  set(ENABLE_GPU_TARGET ON)
endif()

if(${ENABLE_NVIDIA_GPU_BACKEND})
  if(NOT ENABLE_AMD_GPU_BACKEND)
  set(CODEGEN_BACKENDS ${CODEGEN_BACKENDS}nvidia;)
  endif()
  add_definitions(-DENABLE_NVIDIA_GPU_BACKEND)
  set(ENABLE_GPU_TARGET ON)
endif()

if(${ENABLE_FPGA_TARGET})
  add_definitions(-DENABLE_FPGA_TARGET)
endif()

if(${ENABLE_GPU_TARGET})
  add_definitions(-DENABLE_GPU_TARGET)
endif()


#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
  
  find_package(MLIR REQUIRED CONFIG)
  
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(COMET_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

# message(STATUS "BLIS and BLAS configuration: ${CMAKE_BINARY_DIR}/blis")
# message(STATUS "BLIS and BLAS configuration: ${CMAKE_CURRENT_BINARY_DIR}/blis")
#-------------------------------------------------------------------------------
# BLIS Configuration
#-------------------------------------------------------------------------------
  set(BLA_STATIC ON) #linking with BLIS static library
  set(BLA_VENDOR FLAME)
  set(BLA_PREFER_PKGCONFIG TRUE)
  set(BLA_PKGCONFIG_BLAS blis)
  # set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/blis/")
  find_package(BLAS REQUIRED)
  if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS include dirs: ${BLAS_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "BLAS not found!")
  endif()

#-------------------------------------------------------------------------------
# MCL include paths
#-------------------------------------------------------------------------------
if(${ENABLE_FPGA_TARGET})
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/runtimes/mcl/build/include/")
endif()

#-------------------------------------------------------------------------------
# Triton Setup
#-------------------------------------------------------------------------------
if(${ENABLE_GPU_TARGET})
  set(TRITON_PATH "" CACHE PATH "Path to Triton source")
  set(TRITON_BUILD_UT OFF CACHE BOOL "Disable triton unittests" FORCE)
  set(TRITON_CODEGEN_BACKENDS "${CODEGEN_BACKENDS}" CACHE STRING "Enable different codegen backends" FORCE)
  if(NOT DEFINED DEVICE_COMPUTE_CAPABILITY)
    message(FATAL_ERROR "Please specify the default compute capability for your GPU device")
  endif()
  add_compile_definitions(DEVICE_COMPUTE_CAPABILITY="${DEVICE_COMPUTE_CAPABILITY}")

  add_subdirectory(${TRITON_PATH} ${TRITON_BUILD_PATH})
  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
  include_directories("${TRITON_PATH}")
  include_directories("${TRITON_PATH}/include/")
  include_directories("${TRITON_PATH}/third_party/")
  include_directories("${TRITON_BUILD_PATH}/include/")
  include_directories("${TRITON_BUILD_PATH}/third_party/")
endif()

#-------------------------------------------------------------------------------
# COMET configuration
#-------------------------------------------------------------------------------

# COMET project.
set(COMET_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR}/../ ) # --src-root
set(COMET_MAIN_INCLUDE_DIR ${COMET_MAIN_SRC_DIR}/include)

set(COMET_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../)
# set(COMET_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(COMET_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(COMET_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")


# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(comet-headers)
set_target_properties(comet-headers PROPERTIES FOLDER "Misc")
add_custom_target(comet-doc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
message(STATUS "aasaaaaa Using BLAS_INCLUDE_LIBRARIES: ${BLAS_INCLUDE_DIRS}")
# Add MLIR, LLVM and BLIS headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIRS})

# Add COMET files to the include path
include_directories(${COMET_MAIN_INCLUDE_DIR})
include_directories(${COMET_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------
add_subdirectory(${CMAKE_SOURCE_DIR}/../include/comet/ ${CMAKE_BINARY_DIR}/include/comet/)
add_subdirectory(${CMAKE_SOURCE_DIR}/../lib ${CMAKE_BINARY_DIR}/lib)
add_subdirectory(${CMAKE_SOURCE_DIR}/../frontends/comet_dsl ${CMAKE_BINARY_DIR}/frontends/comet_dsl)
add_subdirectory(${CMAKE_SOURCE_DIR}/../test/integration ${CMAKE_BINARY_DIR}/test/integration)


option(COMET_INCLUDE_DOCS "Generate build targets for the COMET docs.")
if (COMET_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()



install(DIRECTORY include/comet
  DESTINATION include
  COMPONENT comet-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${COMET_INCLUDE_DIR}/comet
  DESTINATION include
  COMPONENT comet-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-comet-headers
                           DEPENDS comet-headers
                           COMPONENT comet-headers)
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/install/lib/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set RPATH to $ORIGIN on all targets.
function(set_rpath_all_targets dir)
  get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(subdir ${subdirectories})
    set_rpath_all_targets(${subdir})
  endforeach()

  get_directory_property(LCL_TARGETS DIRECTORY ${dir} BUILDSYSTEM_TARGETS)
  set_property(TARGET ${LCL_TARGETS} PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
endfunction()

option(STANDALONE_INSTALL "Create an 'install' for packaging which doesn't \
         require installation" off)
if (STANDALONE_INSTALL)
  message(STATUS "Setting an $ORIGIN-based RPATH on all executables")
  set_rpath_all_targets(${CMAKE_CURRENT_SOURCE_DIR})
endif()

option(DEBUG_MODE "Create a installation with debug information" off)
if (DEBUG_MODE)
  message(STATUS "Building comet in debug mode")
  add_compile_options(-DCOMET_DEBUG_MODE)
endif()
