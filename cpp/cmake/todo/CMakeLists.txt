cmake_minimum_required ( VERSION 2.8 FATAL_ERROR )

set ( CMAKE_LEGACY_CYGWIN_WIN32 0 )
# ----------------------------------------------------------------------------

project ( todo )

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include (gmock)
# ----------------------------------------------------------------------------

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set (warnings "-Wall -Wextra -Werror")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set (warnings "/W4 /WX /EHsc")
endif()

if (NOT CONFIGURED_ONCE)
    set (CMAKE_CXX_FLAGS "${warnings}"
         CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set (CMAKE_C_FLAGS   "${warnings}"
         CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${warnings}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${warnings}")

# ----------------------------------------------------------------------------

enable_testing ()

# ----------------------------------------------------------------------------

include_directories (${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(todoCore)

add_executable(todo main.cc)
target_link_libraries(todo todoCore)
add_executable (todo main.cc todoCore/todo.cc )
add_test ( todoTest todo )

set (CONFIGURED_ONCE TRUE CACHE INTERNAL
    "A flag showing that CMake has configured at least once.")
