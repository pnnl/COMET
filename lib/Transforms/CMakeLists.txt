add_mlir_dialect_library(COMETTransforms
        AffineCFG.cpp
        InsertMCLCallsPass.cpp
        RaiseToAffine.cpp
        SCFWhileToFor.cpp
        GPUToOCLSPIRVPass.cpp

        ADDITIONAL_HEADER_DIRS
        {$COMET_MAIN_INCLUDE_DIR}/comet/Transforms

        DEPENDS
        CometTransformsIncGen
        )