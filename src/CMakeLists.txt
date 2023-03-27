CPMAddPackage("gh:Neargye/magic_enum@0.7.3")
CPMAddPackage("gh:ericniebler/range-v3#0.12.0")

if (BUILD_DOC AND DOXYGEN_FOUND)
    doxygen_add_docs(docs)
endif ()

add_subdirectory(ast)
add_subdirectory(mlir)
add_subdirectory(compiler)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_sources(voila_compiler PRIVATE
        ${CMAKE_BINARY_DIR}/src/voila_parser.cpp
        ${CMAKE_BINARY_DIR}/src/voila_lexer.cpp
        TypeInferer.cpp
        DotVisualizer.cpp
        MLIRGenerator.cpp
        MLIRGeneratorImpl.cpp
        TypeInferencePass.cpp
        Program.cpp
        Config.cpp
        Types.cpp)

set(LLVM_LINK_COMPONENTS
        LLVMCore
        LLVMSupport
        LLVMTarget
        )

target_link_libraries(voila_compiler
        ${dialect_libs}
        ${conversion_libs}
        magic_enum
        reflex
        voila_ast
        reflex
        range-v3
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRLLVMToLLVMIRTranslation
        MLIROpenMPToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRSupport
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRJitRunner
        MLIRTransforms
        ${LLVM_LINK_COMPONENTS}
        )

target_compile_features(voila_compiler PUBLIC cxx_std_20)
target_compile_options(voila_compiler PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic)

llvm_update_compile_flags(voila_compiler)