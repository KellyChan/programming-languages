#----------------------------------------------------

find_package(OpenCV REQUIRED)

#----------------------------------------------------
# build

# Trackbar
add_executable(Trackbar Trackbar.cpp)
target_link_libraries(Trackbar ${OpenCV_LIBS})

# Video
add_executable(Video Video.cpp)
target_link_libraries(Video ${OpenCV_LIBS})
