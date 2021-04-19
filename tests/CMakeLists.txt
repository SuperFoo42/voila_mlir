CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v1.10.x
    VERSION 1.10.0
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
  )

include(CTest)
enable_testing()
