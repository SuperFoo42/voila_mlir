enable_testing()

add_test(NAME test_tpch COMMAND ./test_tpch.py WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
