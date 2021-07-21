set(LLVM_LINK_COMPONENTS
        Support
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRAnalysis
        MLIRVoila
        MLIRPass
        MLIRSupport
        MLIRIR
        MLIRParser
        MLIRTransforms
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        voila_compiler
        fmt
        reflex
        spdlog
        benchmark::benchmark
        benchmark::benchmark_main
        cereal
        )

add_llvm_executable(tpc-benchmark
        TPCHBenchmark.cpp benchmark_main.cpp
        )

target_link_libraries(tpc-benchmark PRIVATE ${LIBS})

target_compile_features(tpc-benchmark PUBLIC cxx_std_20)
mlir_check_all_link_libraries(tpc-benchmark)