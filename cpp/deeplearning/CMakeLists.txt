CMAKE_MINIMUM_REQUIRED ( VERSION 2.8.8 )

# ----------------------------------------------------------------------------

PROJECT ( FASTERRCNN )

SET ( PROJECT_ROOT ${CMAKE_SOURCE_DIR} )

# ----------------------------------------------------------------------------
# CMake settings

# cmake modules
SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" )

# libraries

# cuda

# caffe
FILE ( COPY ${PROJECT_ROOT}/caffe DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
INCLUDE ( FindCaffe )


# ----------------------------------------------------------------------------
# Sub-directories

ADD_SUBDIRECTORY ( fasterrcnn )

