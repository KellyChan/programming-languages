cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------------
# project

project(OpenCVDemo)

#--------------------------------------------------------
# packages

find_package(OpenCV REQUIRED)

#--------------------------------------------------------
# subdirectories

add_subdirectory(src/core)
add_subdirectory(src/imgproc)
add_subdirectory(src/highgui)
add_subdirectory(src/feature2d)
add_subdirectory(src/gpu)

#--------------------------------------------------------
# build
