CPMAddPackage("gh:Neargye/magic_enum@0.7.2")
CPMAddPackage("gh:jarro2783/cxxopts@2.2.1")
CPMAddPackage(
        NAME fmt
        GIT_TAG 7.1.3
        GITHUB_REPOSITORY fmtlib/fmt
)

if(fmt_ADDED)
    # enable c++11 to avoid compilation errors
    set_target_properties(fmt PROPERTIES CXX_STANDARD 20)
endif()

if (BUILD_DOC AND DOXYGEN_FOUND)
doxygen_add_docs(docs)
endif()

add_subdirectory(ast)

add_library(voila_compiler
voila_parser.cpp
voila_lexer.cpp
)
#TODO: fix libreflex dependency
target_link_libraries(voila_compiler magic_enum reflex fmt voila_ast ${CMAKE_SOURCE_DIR}/../RE-flex/lib/libreflex.so)
target_compile_features(voila_compiler PUBLIC cxx_std_20)
set_property(TARGET voila_compiler PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
target_compile_options(voila_compiler PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Werror
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic)
add_executable(voila
        main.cpp)

target_link_libraries(voila voila_compiler cxxopts fmt reflex ${CMAKE_SOURCE_DIR}/../RE-flex/lib/libreflex.so)
target_compile_options(voila PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Werror
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic)
target_compile_features(voila PUBLIC cxx_std_20)
set_property(TARGET voila PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

