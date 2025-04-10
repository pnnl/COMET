if(ENABLE_NVIDIA_GPU_BACKEND)
  find_package(CUDAToolkit)
  if(NOT ${CUDAToolkit_FOUND})
      message(WARNING "CUDAToolkit not found. You can still lower to GPU Assembly code (use --gpu-code-format=Assembly), but you will not be able to run it using COMET's execution engine")
  endif()
endif()

if(ENABLE_AMD_GPU_BACKEND)
  find_package(hip)
  if(NOT ${hip_FOUND})
      message(WARNING "Hip not found. Unable to use GPU target")
  endif()
endif()

set(SOURCES
  blis_interface.cpp
  StatUtils.cpp
  SparseUtils.cpp
  TransposeUtils.cpp
  
)

if(ENABLE_NVIDIA_GPU_BACKEND)
  if(${CUDAToolkit_FOUND})
    set(SOURCES ${SOURCES} CuGpuUtils.cpp)
  endif()
endif()

if(ENABLE_AMD_GPU_BACKEND)
  if(${hip_FOUND})
    set(SOURCES ${SOURCES} HipGpuUtils.cpp)
  endif()
endif()

add_llvm_library(comet_runner_utils
  PARTIAL_SOURCES_INTENDED
  SHARED
  ${SOURCES}
)

target_compile_definitions(comet_runner_utils PRIVATE comet_runner_utils_EXPORTS comet_blis_interface_EXPORTS)
set(LIBS
  ${BLAS_LIBRARIES} 
  LLVMSupport
)

if(ENABLE_NVIDIA_GPU_BACKEND)
  if(${CUDAToolkit_FOUND})
    set(LIBS ${LIBS} CUDA::cuda_driver)
    set(LIBS ${LIBS} CUDA::cudart)
  endif()
endif()

if(ENABLE_AMD_GPU_BACKEND)
  if(${hip_FOUND})
    set(LIBS ${LIBS} hip::host)
  endif()
endif()

target_link_libraries(comet_runner_utils ${LIBS})