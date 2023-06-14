target_sources(voila_ast PRIVATE
        ${CMAKE_BINARY_DIR}/src/voila_parser.cpp
        Assign.cpp
        Emit.cpp
        Fun.cpp
        FunctionCall.cpp
        Gather.cpp
        Hash.cpp
        Insert.cpp
        Loop.cpp
        Lookup.cpp
        Scatter.cpp
        Selection.cpp
        Variable.cpp
        Not.cpp
        Read.cpp
        Ref.cpp
        StatementWrapper.cpp
        Write.cpp
        Predicate.cpp
        Load.cpp)

target_link_libraries(voila_ast magic_enum range-v3)
target_compile_features(voila_ast PUBLIC cxx_std_20)
target_compile_options(voila_ast PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic)