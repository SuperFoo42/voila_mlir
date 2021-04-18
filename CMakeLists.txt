cmake_minimum_required(VERSION 3.11)

project(voila_mlir)

# Download gtest and unittest submodule
find_package(Git)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --remote
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/tests/extern/gtest/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif ()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${PROJECT_SOURCE_DIR}/tests/extern/gtest
        ${CMAKE_CURRENT_BINARY_DIR}/gtest-build
        EXCLUDE_FROM_ALL)

include(CTest)
enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_FLAGS " ${ARCH_FLAGS} -g -fPIC -Wall -Wextra --std=c++17")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}  ${ARCH_FLAGS} -fno-omit-frame-pointer -fsanitize=address  -DIS_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ARCH_FLAGS} -fno-omit-frame-pointer -fsanitize=address  -DIS_DEBUG")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ARCH_FLAGS} -O3 -g -fno-omit-frame-pointer -DIS_RELEASE")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ARCH_FLAGS} -O3 -g -fno-omit-frame-pointer -DIS_RELEASE")
set(CMAKE_LINKER_FLAGS_RELEASE "${CMAKE_LINKER_FLAGS_RELEASE} ${ARCH_FLAGS} -O3 -g -fno-omit-frame-pointer -DIS_RELEASE")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(TBB)
include_directories(${TBB_INCLUDE_DIRS})

add_definitions(${TBB_DEFINITIONS})

find_package(BISON)
find_package(FLEX)

BISON_TARGET(VoilaParser voila.y ${CMAKE_CURRENT_SOURCE_DIR}/src/voila_parser.cpp
             DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/include/voila_parser.hpp)
			 
FLEX_TARGET(VoilaScanner lexer.l  ${CMAKE_CURRENT_SOURCE_DIR}/src/voila_lexer.cpp)

ADD_FLEX_BISON_DEPENDENCY(VoilaScanner VoilaParser)

include(CheckSymbolExists)

CHECK_SYMBOL_EXISTS(mmap "sys/mman.h" HAVE_POSIX_MMAP)
CHECK_SYMBOL_EXISTS(mremap "sys/mman.h" HAVE_LINUX_MREMAP)
CHECK_SYMBOL_EXISTS(sysconf "unistd.h" HAVE_POSIX_SYSCONF)
CHECK_SYMBOL_EXISTS(MAP_POPULATE "sys/mman.h" HAVE_LINUX_MAP_POPULATE)

find_package(Threads REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(tests)