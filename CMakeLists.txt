cmake_minimum_required(VERSION 3.11)

project(voila_mlir)

include(cmake/CPM.cmake)

option(BUILD_DOC "Build documentation" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_BENCH "Build benchmarks" ON)

if (BUILD_DOC)
find_package(Doxygen
             REQUIRED dot
             OPTIONAL_COMPONENTS mscgen dia)
endif()
#Enable sanitizers globally
add_compile_options(-fsanitize=address -fsanitize=undefined)
add_link_options(-fsanitize=address -fsanitize=undefined)

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
  GIT_TAG v3.0.3
)
if(reflex_ADDED)
  add_library(reflex INTERFACE IMPORTED)
  target_include_directories(reflex INTERFACE "${reflex_SOURCE_DIR}/include")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
if (BUILD_TEST)
add_subdirectory(tests)
endif()
if (BUILD_BENCH)
add_subdirectory(benchmark)
endif()