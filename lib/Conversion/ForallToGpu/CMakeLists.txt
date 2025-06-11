add_llvm_library(COMETForallToGpu
    ForallToGpu.cpp

    DEPENDS
    ForallConversionPassIncGen

    # LINK_LIBS
    # MLIRPASS
) 