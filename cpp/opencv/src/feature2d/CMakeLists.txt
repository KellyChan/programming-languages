#-----------------------------------------------------

find_package(OpenCV REQUIRED)

#-----------------------------------------------------

# HarrisDetector
add_executable(CornerHarrisDetector CornerHarrisDetector.cpp)
target_link_libraries(CornerHarrisDetector ${OpenCV_LIBS})

# ShiTomasiDetector
add_executable(CornerShiTomasiDetector CornerShiTomasiDetector.cpp)
target_link_libraries(CornerShiTomasiDetector ${OpenCV_LIBS})

# CustomDetector
add_executable(CornerCustomDetector CornerCustomDetector.cpp)
target_link_libraries(CornerCustomDetector ${OpenCV_LIBS})

# SubPixelesDetector
add_executable(CornerSubPixelesDetector CornerSubPixelesDetector.cpp)
target_link_libraries(CornerSubPixelesDetector ${OpenCV_LIBS})

# FeatureDetector
# add_executable(FeatureDetector FeatureDetector.cpp)
# target_link_libraries(FeatureDetector ${OpenCV_LIBS})
