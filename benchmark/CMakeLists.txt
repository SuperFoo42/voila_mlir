CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.6.0
        OPTIONS "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}" "BENCHMARK_ENABLE_TESTING Off" "BENCHMARK_ENABLE_LTO True"
)
if (benchmark_ADDED)
    # enable c++11 to avoid compilation errors
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)
    target_compile_options(benchmark PUBLIC -Wno-suggest-override)
endif ()

CPMAddPackage(
        NAME cereal
        GIT_TAG master
        GITHUB_REPOSITORY USCiLab/cereal
        OPTIONS "SKIP_PORTABILITY_TEST ON" "JUST_INSTALL_CEREAL ON"
)

CPMAddPackage("gh:vincentlaucsb/csv-parser#2.1.3")

CPMAddPackage("gh:Cyan4973/xxHash#v0.8.1")

CPMAddPackage(
        NAME bxzstr
        GIT_TAG master
        GITHUB_REPOSITORY tmaklin/bxzstr
        OPTIONS "ZLIB_FOUND FALSE" "BZIP2_FOUND FALSE"
)

if (bxzstr_ADDED)
    add_library(bxzstr INTERFACE IMPORTED)
    target_include_directories(bxzstr INTERFACE "${bxzstr_SOURCE_DIR}/include")
    target_link_libraries(bxzstr INTERFACE lzma)
endif ()

#TODO
#CPMAddPackage("gh:xqhe/db-engine-paradigms#master")

include_directories(include)
add_subdirectory(src)

add_library(benchmark_headers INTERFACE)
set_target_properties(benchmark_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
        )
target_link_libraries(benchmark_headers INTERFACE cereal bxzstr)
configure_file(benchmark_defs.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark_defs.hpp.inc)