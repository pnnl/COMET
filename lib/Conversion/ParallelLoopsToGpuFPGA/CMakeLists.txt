add_llvm_library(COMETParallelLoopsToGpuFPGA
    ParallelLoopsToGpuFPGA.cpp

    DEPENDS
    ParallelLoopsToGpuFPGAConversionPassIncGen

    # LINK_LIBS
    # MLIRPASS
) 