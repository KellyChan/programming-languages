set (GMOCK_DIR "../../../../gmock/gmock-1.6.0"
     CACHE PATH "THe path to the GoogleMock test framework.")


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # force this option to ON so that Google Test will use /MD instead of /MT
    # /MD is now the default for cmake, so it should be our default, too
    option (gtest_force_shared_crt
            "Use shared (DLL) run-time lib even when Google Test is built as static lib."
            ON)
elseif (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1)
endif()


add_subdirectory(${GMOCK_DIR} ${CMAKE_BINARY_DIR}/gmock)
set_property(TARGET gtest APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
include_directories (SYSTEM ${GMOCK_DIR}/gtest/include
                            ${GMOCK_DIR}/include)


# add_executable (todoTest todoTest.cc)
# target_link_libraries (todoTest todoCore gmock_main )
# add_test(todoTest todoTest)

#
# add_gmock_test(<target> <sources>...)
#
#  Adds a Google Mock based test executable, <target>, built from <sources> and
#  adds the test so that CTest will run it. Both the executable and the test
#  will be named <target>.
#
# function(add_gmock_test target)
#    add_executable(${target} ${ARGN})
#    target_link_libraries(${target} gmock_main)
#    add_test(${target} ${target})
# endfunction()


add_gmock_test(todoTest todoTest.cc)
target_link_libraries(todoTest todoCore)
