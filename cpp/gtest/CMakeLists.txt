CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

# ----------------------------------------------------------------------------
# Project settings

SET (PROJECT_NAME gtestdemo)
PROJECT (${PROJECT_NAME})

SET (PROJECT_ROOT ${CMAKE_SOURCE_DIR})
# ----------------------------------------------------------------------------
# Library settings


# ----------------------------------------------------------------------------
# Compiler settings

SET (CMAKE_CXX_FLAGS "-g -Wall")

OPTION (test "Build all tests." ON)

# ----------------------------------------------------------------------------
# Executable settings


ADD_SUBDIRECTORY ( demo )

# ----------------------------------------------------------------------------
# Option: test

if (test)

    # config gtest
    ADD_SUBDIRECTORY (demo/lib/googletest)
    ENABLE_TESTING()
    INCLUDE_DIRECTORIES (${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

    # executable configurations
    ADD_EXECUTABLE (runTests demo/foo/tests/test_foo.cpp)
    TARGET_LINK_LIBRARIES (runTests gtest gtest_main pthread)

    # 'make test' configuration
    # ADD_TEST (NAME xxx COMMAND runTests)
    # ADD_TEST (xxx runTests)

endif()
