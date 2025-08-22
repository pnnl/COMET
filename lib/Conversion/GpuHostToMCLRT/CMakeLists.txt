add_llvm_library(COMETGpuHostToMCLRT
    GpuHostToMCLRTPass.cpp

    DEPENDS
    GpuHostToMCLRTConversionPassIncGen
) 