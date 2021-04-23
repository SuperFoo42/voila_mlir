CPMAddPackage("gh:Neargye/magic_enum@0.7.2")
CPMAddPackage("gh:jarro2783/cxxopts@2.2.1")
CPMAddPackage("gh:fmtlib/fmt#7.1.3")

if (BUILD_DOC AND DOXYGEN_FOUND)
doxygen_add_docs(docs)
endif()

add_subdirectory(ast)

add_library(voila_compiler
${BISON_VoilaParser_OUTPUTS}
${FLEX_VoilaScanner_OUTPUTS}
)

target_link_libraries(voila_compiler magic_enum reflex fmt voila_ast)
target_compile_features(voila_compiler PUBLIC cxx_std_20)