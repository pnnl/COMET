add_llvm_tool(comet-runner
        comet-runner.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_update_compile_flags(comet-runner)

set(LIBS
        MLIRAnalysis
        MLIRAffineDialect
        MLIRDialect
        MLIRExecutionEngine
        MLIRFuncDialect
        MLIRLinalgToLLVM
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIRMemRefToLLVM
        MLIRSCFDialect
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRSupport
        MLIRSideEffectInterfaces
        MLIRJitRunner
        COMETTransforms
        mlir_c_runner_utils
        mlir_runner_utils
        comet_runner_utils
        mcl
        )

target_link_libraries(comet-runner
        PRIVATE MLIRIR
        ${LIBS}
        ${dialect_libs}
        ${conversion_libs}
        )