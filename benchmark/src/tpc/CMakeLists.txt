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
        MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        voila_compiler
        reflex
        spdlog
        benchmark::benchmark
        benchmark::benchmark_main
        xxhash
        benchmark_utils
        )

add_executable(tpc-benchmark
        TPCHBenchmark.cpp
        benchmark_main.cpp
        BenchmarkState.cpp
        )

target_link_libraries(tpc-benchmark PRIVATE ${LIBS})
target_compile_options(tpc-benchmark PUBLIC
        -Wpedantic
        -Wall
        -Werror
        -Wextra
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic
        -Wno-suggest-override)
#[[if ($<CONFIG:Release>)
    target_compile_options(tpc-benchmark PUBLIC -O3)
endif ()]]

target_compile_features(tpc-benchmark PUBLIC cxx_std_20)
#mlir_check_all_link_libraries(tpc-benchmark)