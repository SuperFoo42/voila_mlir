#[[
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

]]

llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(VOILA_OPT_TEST_DEPENDS
        FileCheck count not
        voila-opt
        voila-mlir-translate
        )

if(MLIR_ENABLE_BINDINGS_PYTHON)
    list(APPEND VOILA_OPT_TEST_DEPENDS VoilaPythonModules)
endif()

add_lit_testsuite(check-voila-opt "Running the voila-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${VOILA_OPT_TEST_DEPENDS}
        )

add_lit_testsuite(check-voila-translate "Running the voila-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${VOILA_OPT_TEST_DEPENDS}
        )

set_target_properties(check-voila-translate PROPERTIES FOLDER "voila/translation")

set_target_properties(check-voila-opt PROPERTIES FOLDER "voila/lowering")

add_lit_testsuites(VOILA_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${VOILA_OPT_TEST_DEPENDS})