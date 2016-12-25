cmake_minimum_required ( VERSION 2.8 FATAL_ERROR )

set ( CMAKE_LEGACY_CYGWIN_WIN32 0 )
# ----------------------------------------------------------------------------

project ( todo )

enable_testing ()

add_executable (todo main.cc todo.cc )
add_test (todoTest todo )

