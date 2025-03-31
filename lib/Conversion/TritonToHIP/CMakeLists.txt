add_llvm_library(COMETTritonToHIP
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
)
