CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

# ----------------------------------------------------------------------------
# Project settings

PROJECT ( MUSE CXX C )
SET ( MUSE_MAJOR_VERSION 0 )
SET ( MUSE_MINOR_VERSION 0 )
SET ( MUST_PATCH_VERSION 1 )
SET ( MUST_VERSION ${MUSE_MAJOR_VERSION}.${MUSE_MINOR_VERSION}.${MUSE_PATCH_VERSION} )

# ----------------------------------------------------------------------------
# CMkae settings

SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" )
SET ( PROJECT_ROOT ${CMAKE_SOURCE_DIR} )

# include cmake package

# ----------------------------------------------------------------------------
# Sub-directory settings

INCLUDE_DIRECTORIES ( "${PROJECT_ROOT}" )

# static files
# FILE ( COPY ${PROJECT_ROOT}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# modules
ADD_SUBDIRECTORY ( muse )
