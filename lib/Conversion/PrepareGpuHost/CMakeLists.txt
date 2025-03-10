add_llvm_library(COMETPrepareGpuHost
    PrepareGpuHost.cpp

    DEPENDS
    PrepareGpuHostPassIncGen

    LINK_LIBS PUBLIC
)
