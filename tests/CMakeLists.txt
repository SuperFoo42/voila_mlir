configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(VOILA_TEST_DEPENDS
        FileCheck count not
        voila
        )

add_lit_testsuite(check-voila "Running the standalone regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${VOILA_TEST_DEPENDS}
        )
set_target_properties(check-voila PROPERTIES FOLDER "Tests")

#add_lit_testsuites(VOILA ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${VOILA_TEST_DEPENDS})
