# cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------------
# project

# project(OpenCVDemo)

#--------------------------------------------------------
# packages

find_package(OpenCV REQUIRED)

#--------------------------------------------------------
# build

# DisplayImage
add_executable(DisplayImage DisplayImage.cpp)
target_link_libraries(DisplayImage ${OpenCV_LIBS})

# ModifyImage
add_executable(ModifyImage ModifyImage.cpp)
target_link_libraries(ModifyImage ${OpenCV_LIBS})

# ScanImage
add_executable(ScanImage ScanImage.cpp)
target_link_libraries(ScanImage ${OpenCV_LIBS})

# Mask
add_executable(Mask Mask.cpp)
target_link_libraries(Mask ${OpenCV_LIBS})

# Weighted
add_executable(Weighted Weighted.cpp)
target_link_libraries(Weighted ${OpenCV_LIBS})

# Brightness
add_executable(Brightness Brightness.cpp)
target_link_libraries(Brightness ${OpenCV_LIBS})

# Drawing
add_executable(Drawing Drawing.cpp)
target_link_libraries(Drawing ${OpenCV_LIBS})
