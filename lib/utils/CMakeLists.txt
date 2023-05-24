add_llvm_library(COMETStats
  dummyStats.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/utils
  )

  target_link_libraries(COMETStats MLIRIR)