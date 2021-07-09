cmake_minimum_required(VERSION 3.14)

project(voila_mlir LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

include(cmake/CPM.cmake)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
option(BUILD_DOC "Build documentation" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_BENCH "Build benchmarks" ON)

# check LTO support
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORT OUTPUT error)
#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION LTO_SUPPORT)

# check PIE support
include(CheckPIESupported)
check_pie_supported(OUTPUT_VARIABLE PIE_SUPPORT LANGUAGES C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE PIE_SUPPORT)

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
        COMMAND ${CMAKE_COMMAND} -B ${llvm_BINARY_DIR} -G Ninja -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=True -DLLVM_ENABLE_LLD=True -DLLVM_ENABLE_EH=True -DLLVM_ENABLE_BINDINGS=False -DLLVM_CCACHE_BUILD=True -DLLVM_BUILD_TESTS=True
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

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

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
            REQUIRED dot
            OPTIONAL_COMPONENTS mscgen dia)
endif ()

#Enable sanitizers globally
add_compile_options(-fsanitize=address -fsanitize=undefined)
add_link_options(-fuse-ld=lld -fsanitize=address -fsanitize=undefined) #need to use lld, for some reason ld complains about invalid symbols.

find_package(TBB)
include_directories(${TBB_INCLUDE_DIRS})

add_definitions(${TBB_DEFINITIONS})

include(CheckSymbolExists)

find_package(Threads REQUIRED)

find_package(BISON REQUIRED)

#TODO: add reflex call to generate lexer for grammar
CPMAddPackage(
        NAME reflex
        GITHUB_REPOSITORY Genivia/RE-flex
        GIT_TAG v3.0.7
)
if (reflex_ADDED)
    execute_process(
            COMMAND sh ./clean.sh
            WORKING_DIRECTORY ${reflex_SOURCE_DIR}
    )
    execute_process(COMMAND sh ./build.sh
            WORKING_DIRECTORY ${reflex_SOURCE_DIR}
            )
    add_library(reflex INTERFACE IMPORTED include/mlir/lowering/LogicalOpLowering.hpp include/mlir/lowering/MoveOpLowering.cpp include/mlir/lowering/MoveOpLowering.hpp include/ast/Hash.cpp include/ast/Hash.hpp)
    target_include_directories(reflex INTERFACE "${reflex_SOURCE_DIR}/include")
    target_link_libraries(reflex INTERFACE "${reflex_SOURCE_DIR}/lib/libreflex.a")
endif ()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(include)
if (BUILD_TEST)
    add_subdirectory(tests)
endif ()
if (BUILD_BENCH)
    add_subdirectory(benchmark)
endif ()