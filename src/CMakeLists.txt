CPMAddPackage("gh:Neargye/magic_enum@0.7.3")
CPMAddPackage("gh:jarro2783/cxxopts@3.0.0")
CPMAddPackage("gh:fmtlib/fmt#8.1.0")

CPMAddPackage(
        NAME spdlog
        GIT_TAG v1.9.2
        GITHUB_REPOSITORY gabime/spdlog
        OPTIONS "SPDLOG_FMT_EXTERNAL ON" #we already use our own fmt
)

CPMAddPackage("gh:ericniebler/range-v3#0.11.0")


if (fmt_ADDED)
    # enable c++20 to avoid compilation errors
    set_target_properties(fmt PROPERTIES CXX_STANDARD 20)
endif ()

if (BUILD_DOC AND DOXYGEN_FOUND)
    doxygen_add_docs(docs)
endif ()

add_subdirectory(ast)
add_subdirectory(mlir)
add_subdirectory(compiler)

#FIXME
#add_custom_command(
#        DEPENDS ${CMAKE_SOURCE_DIR}/voila.y
#        COMMAND ${BISON_EXECUTABLE} -d -t ${CMAKE_SOURCE_DIR}/voila.y
#        OUTPUT ${CMAKE_SOURCE_DIR}/src/voila_parser.cpp ${CMAKE_SOURCE_DIR}/include/voila_parser.hpp ${CMAKE_SOURCE_DIR}/include/location.hpp
#        )

#add_custom_command(
#        DEPENDS ${CMAKE_SOURCE_DIR}/voila.lex
#        COMMAND ${reflex_SOURCE_DIR}/bin/reflex -d --fast --bison-complete --bison-locations --namespace=voila::lexer --lexer=Lexer --header-file=${CMAKE_SOURCE_DIR}/include/voila_lexer.hpp −−graphs-file=lexer.gv -o ${CMAKE_SOURCE_DIR}/src/voila_lexer.cpp ${CMAKE_SOURCE_DIR}/voila.lex
#        OUTPUT ${CMAKE_SOURCE_DIR}/src/voila_lexer.cpp ${CMAKE_SOURCE_DIR}/include/voila_lexer.hpp
#)

add_library(voila_compiler
        voila_parser.cpp
        voila_lexer.cpp
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
        voila_lowering
        magic_enum
        reflex
        voila_ast
        reflex
        cxxopts
        spdlog
        range-v3
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRStandard
        MLIRLLVMToLLVMIRTranslation
        MLIROpenMPToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRStandardToLLVM
        MLIRSupport
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRJitRunner
        MLIRTransforms
        MLIRTranslation
        ${LLVM_LINK_COMPONENTS}
        papi
        )

target_compile_features(voila_compiler PUBLIC cxx_std_20)
target_compile_options(voila_compiler PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Werror
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic)

