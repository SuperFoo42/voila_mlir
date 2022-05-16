#CPMAddPackage("gh:google/googletest#release-1.11.0")

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
        cxxopts
        reflex
        spdlog
        gtest
        gtest_main
        gmock
        xxhash
        benchmark_headers
        benchmark_utils
        )

add_llvm_executable(voila_tests
        main.cpp
        VoilaOpTests.cpp
        TPCBenchmarkTests.cpp
        )

target_compile_features(voila_tests PUBLIC cxx_std_20)

target_compile_options(voila_tests PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic
        -Wno-suggest-override
        -Wno-covered-switch-default)

target_link_libraries(voila_tests PRIVATE ${LIBS})

mlir_check_all_link_libraries(voila_tests)

enable_testing()
add_test(VoilaTests PUBLIC voila_tests)

configure_file(test_defs.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/test_defs.hpp.inc)