add_llvm_library(COMETTritonToHIP
    # GpuToTritonConversion.cpp
    TritonToHIPPass.cpp

    DEPENDS
    TritonDeviceToHIPConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonAMDGPUTransforms
    AMDGPUIR
)
