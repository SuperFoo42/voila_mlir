cmake_minimum_required(VERSION 3.15)

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()


project(voila_mlir LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

set(CMAKE_BUILD_TYPE DEBUG)
#Enable sanitizers globally
set(DEBUG_BUILD -O0 -g -Wno-ambiguous-reversed-operator -Wno-covered-switch-default -Wno-unused-parameter)
set(RELEASE_BUILD -O3 -g -march=native -mavx2 -Wno-ambiguous-reversed-operator -Wno-unused-function -Wno-covered-switch-default -Wno-unused-parameter)
add_compile_options(${DEBUG_BUILD}
 #      "$<$<CONFIG:Debug>:${DEBUG_BUILD}>"
#        "$<$<CONFIG:Release>:${RELEASE_BUILD}>"
)

add_link_options(-fuse-ld=mold)
#add_link_options("$<$<CONFIG:DEBUG>:-fsanitize=address>")

include(cmake/CPM.cmake)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
option(BUILD_DOC "Build documentation" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_BENCH "Build benchmarks" ON)

# check PIE support
if (NOT CMAKE_POSITION_INDEPENDENT_CODE)
include(CheckPIESupported)
check_pie_supported(OUTPUT_VARIABLE PIE_SUPPORT LANGUAGES C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE PIE_SUPPORT)
endif ()

if (NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORT OUTPUT error)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION IPO_SUPPORT)
endif ()

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

CPMAddPackage(
        NAME llvm
        GITHUB_REPOSITORY llvm/llvm-project
        GIT_TAG main
        DOWNLOAD_ONLY
)

execute_process(
        COMMAND ${CMAKE_COMMAND} -B ${llvm_BINARY_DIR} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -G ${CMAKE_GENERATOR} -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=True
                -DLLVM_USE_LINKER=mold -DLLVM_ENABLE_EH=True -DLLVM_ENABLE_BINDINGS=False
                -DLLVM_CCACHE_BUILD=True -DLLVM_BUILD_TESTS=False -DLLVM_ENABLE_IDE=TRUE -DLLVM_OPTIMIZED_TABLEGEN=TRUE
                -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_LTO=OFF # ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}
                -DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV
                -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_PARALLEL_LINK_JOBS=1
        WORKING_DIRECTORY ${llvm_SOURCE_DIR}/llvm
)
message(STATUS "build llvm")
execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${llvm_BINARY_DIR}
)

set(MLIR_DIR "${llvm_BINARY_DIR}/lib/cmake/mlir/")
set(LLVM_EXTERNAL_LIT "${llvm_BINARY_DIR}/build/bin/llvm-lit")
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${llvm_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${llvm_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
# just hope, the first entry is the correct one...
list(GET OpenMP_C_LIBRARIES 0 OpenMP_MLIR_LIB)

configure_file(defs.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/include/defs.hpp)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#enable rtti and exceptions for our project
set(LLVM_REQUIRES_RTTI 1)
set(LLVM_ENABLE_EH 1)
set(LLVM_ENABLE_RTTI True)
set(LLVM_ENABLE_LTO "On")

if (BUILD_DOC)
    find_package(Doxygen
            OPTIONAL_COMPONENTS mscgen dia)
endif ()



include(CheckSymbolExists)

find_package(Threads REQUIRED)

find_package(BISON REQUIRED)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/src/voila_parser.cpp
        COMMAND ${BISON_EXECUTABLE} -t ${CMAKE_SOURCE_DIR}/voila.y --header=${CMAKE_BINARY_DIR}/include/voila_parser.hpp --output=${CMAKE_BINARY_DIR}/src/voila_parser.cpp -Dapi.location.file='"${CMAKE_BINARY_DIR}/include/location.hpp"'
        COMMENT "Generating Voila parser"
)

CPMAddPackage(
        NAME reflex
        GITHUB_REPOSITORY Genivia/RE-flex
        GIT_TAG v3.5.1
        DOWNLOAD_ONLY
)

execute_process(
        COMMAND ${CMAKE_COMMAND} -B ${reflex_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${reflex_BINARY_DIR}/install
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -G ${CMAKE_GENERATOR} ${reflex_SOURCE_DIR}
)
message(STATUS "building and installing Reflex")
execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${reflex_BINARY_DIR} --target install
)

set(Reflex_DIR "${reflex_BINARY_DIR}/install/lib64/cmake/reflex")
find_package(Reflex REQUIRED)
#include_directories(${reflex_BINARY_DIR}/install/include)
#if (reflex_ADDED)
#    execute_process(
#            COMMAND sh ./clean.sh
#            WORKING_DIRECTORY ${reflex_SOURCE_DIR}
#    )
#    execute_process(COMMAND sh ./build.sh
#            WORKING_DIRECTORY ${reflex_SOURCE_DIR}
#            )
    #add_library(reflex INTERFACE IMPORTED include/mlir/Dialects/Voila/lowering/LogicalOpLowering.hpp include/ast/Hash.cpp include/ast/Hash.hpp)
    #target_include_directories(reflex INTERFACE "${reflex_SOURCE_DIR}/include")

#    cmake_policy(SET CMP0004 OLD) # workaround for underscore in reflex lib path
#    target_link_libraries(reflex INTERFACE "${reflex_SOURCE_DIR}/lib/libreflexmin.a")
#endif ()

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/src/voila_lexer.cpp
        COMMAND Reflex::Reflex -d --fast --bison-complete --bison-locations --namespace=voila::lexer --lexer=Lexer --header-file=${CMAKE_BINARY_DIR}/include/voila_lexer.hpp −−graphs-file=${CMAKE_BINARY_DIR}/lexer.gv -o ${CMAKE_BINARY_DIR}/src/voila_lexer.cpp ${CMAKE_SOURCE_DIR}/voila.lex
        DEPENDS ${CMAKE_BINARY_DIR}/src/voila_parser.cpp
)

#include_directories(${reflex_BINARY_DIR}/install/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_custom_target(voila_parser_and_lexer ALL
        DEPENDS ${CMAKE_BINARY_DIR}/src/voila_lexer.cpp ${CMAKE_BINARY_DIR}/src/voila_parser.cpp
        SOURCES ${CMAKE_BINARY_DIR}/src/voila_lexer.cpp ${CMAKE_BINARY_DIR}/src/voila_parser.cpp
        COMMENT "Generating Voila lexer and parser"
        )

# add targets
add_library(voila_compiler)
add_library(voila_ast)

add_dependencies(voila_compiler voila_parser_and_lexer)
add_dependencies(voila_ast voila_parser_and_lexer)

add_subdirectory(src)

if(MLIR_ENABLE_BINDINGS_PYTHON)
    message(STATUS "Enabling Python API")
    add_subdirectory(python)
endif()

add_subdirectory(include)
if (BUILD_TEST)
    CPMAddPackage(
            NAME googletest
            GITHUB_REPOSITORY google/googletest
            GIT_TAG v1.14.0
            VERSION 1.14.0
            OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
    )
    add_subdirectory(tests)
endif ()
if (BUILD_BENCH)
    add_subdirectory(benchmark)
endif ()
