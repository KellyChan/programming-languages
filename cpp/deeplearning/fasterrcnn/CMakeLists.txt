# ----------------------------------------------------------------------------
#

FILE ( GLOB FASTERRCNN_SOURCES . main.cpp )

ADD_EXECUTABLE ( main main.cpp )
TARGET_LINK_LIBRARIES ( main )
