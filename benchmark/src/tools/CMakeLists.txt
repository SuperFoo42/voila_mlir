add_library(benchmark_utils
        DateReformatter.cpp
        DictionaryCompressor.cpp
        Tables.cpp)


target_link_libraries(benchmark_utils
        rapidcsv
        cereal
        cxxopts
        magic_enum
        range-v3
        bxzstr
        )

target_compile_features(benchmark_utils PUBLIC cxx_std_20)
target_compile_options(benchmark_utils PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Werror
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-suggest-override
        -Wno-c++98-compat-pedantic)


add_executable(preprocessor TablePreprocessor.cpp)

target_link_libraries(preprocessor benchmark_utils)
target_compile_features(preprocessor PUBLIC cxx_std_20)