add_llvm_library(COMETGpuToOCLSPIRV
    GpuToOCLSPIRVPass.cpp

    DEPENDS
    GpuToOCLSPIRVConversionPassIncGen
) 