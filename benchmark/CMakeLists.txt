CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.5.6
        OPTIONS "BENCHMARK_ENABLE_TESTING Off"
        OPTIONS "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}"
)
CPMAddPackage(
        NAME cereal
        VERSION 1.3.0
        GITHUB_REPOSITORY USCiLab/cereal
        OPTIONS "SKIP_PORTABILITY_TEST ON" "JUST_INSTALL_CEREAL ON"
)
CPMAddPackage("gh:d99kris/rapidcsv#v8.53")

CPMAddPackage("gh:Cyan4973/xxHash#v0.8.0")

if (benchmark_ADDED)
    # enable c++11 to avoid compilation errors
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)
    target_compile_options(benchmark PUBLIC -Wno-suggest-override)
endif ()
include_directories(include)
add_subdirectory(src)

configure_file(benchmark_defs.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark_defs.hpp.inc)