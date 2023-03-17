llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(VOILA_OPT_TEST_DEPENDS
        FileCheck count not
        voila-opt
        )

set(VOILA_TRANSLATE_TEST_DEPENDS
        FileCheck count not
        voila-mlir-translate
        )

if(MLIR_ENABLE_BINDINGS_PYTHON)
    list(APPEND VOILA_OPT_TEST_DEPENDS VoilaPythonModules)
    list(APPEND VOILA_TRANSLATE_TEST_DEPENDS VoilaPythonModules)
endif()

add_lit_testsuite(check-voila-opt "Running the voila-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${VOILA_OPT_TEST_DEPENDS}
        )

add_lit_testsuite(check-voila-translate "Running the voila-translate regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${VOILA_TRANSLATE_TEST_DEPENDS}
        )

set_target_properties(check-voila-translate PROPERTIES FOLDER "translation")

set_target_properties(check-voila-opt PROPERTIES FOLDER "lowering")

add_lit_testsuites(VOILA_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${VOILA_OPT_TEST_DEPENDS})

add_lit_testsuites(VOILA_TRANSLATE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${VOILA_TRANSLATE_TEST_DEPENDS})