add_llvm_library(COMETGpuToOCLSPIRV
    GpuToOCLSPIRVPass.cpp
    GPUToSPIRVPass.cpp

    DEPENDS
    GpuToOCLSPIRVConversionPassIncGen
) 