add_executable(preprocessor
        TablePreprocessor.cpp
        DateReformatter.cpp
        DictionaryCompressor.cpp)

target_link_libraries(preprocessor rapidcsv cereal cxxopts magic_enum range-v3)
target_compile_features(preprocessor PUBLIC cxx_std_20)