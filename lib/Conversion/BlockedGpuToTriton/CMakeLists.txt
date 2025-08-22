add_llvm_library(COMETBlockedGpuToTriton
    BlockedGpuToTriton.cpp
    BlockedGpuToTritonConversion.cpp

    DEPENDS
    BlockedGpuToTritonPassIncGen
) 