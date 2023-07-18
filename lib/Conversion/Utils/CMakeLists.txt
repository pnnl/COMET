add_llvm_library(COMETGPUUtils
  GPUUtils.cpp
  MarkerUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/Utils
  )
