CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.5.2
        OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)
CPMAddPackage(
        NAME cereal
        VERSION 1.3.0
        GITHUB_REPOSITORY USCiLab/cereal
        OPTIONS "SKIP_PORTABILITY_TEST ON" "JUST_INSTALL_CEREAL ON"
)
CPMAddPackage("gh:d99kris/rapidcsv#v8.51")

if (benchmark_ADDED)
    # enable c++11 to avoid compilation errors
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)
endif ()
include_directories(include)
add_subdirectory(src)

configure_file(benchmark_defs.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark_defs.hpp.inc)