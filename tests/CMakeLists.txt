#CPMAddPackage("gh:google/googletest#release-1.11.0")

add_executable(voila_tests main.cpp)
target_link_libraries(voila_tests gtest gtest_main gmock)

enable_testing()
add_test(VoilaTests PUBLIC voila_tests)