#---------------------------------------------------------
# packages

find_package(OpenCV REQUIRED)

#---------------------------------------------------------
# build

# Smoothing
add_executable(Smoothing Smoothing.cpp)
target_link_libraries(Smoothing ${OpenCV_LIBS})

# Erosion
add_executable(Erosion Erosion.cpp)
target_link_libraries(Erosion ${OpenCV_LIBS})

# Transformation
add_executable(Transformation Transformation.cpp)
target_link_libraries(Transformation ${OpenCV_LIBS})

# Pyramids
add_executable(Pyramids Pyramids.cpp)
target_link_libraries(Pyramids ${OpenCV_LIBS})

# Threshold
add_executable(Threshold Threshold.cpp)
target_link_libraries(Threshold ${OpenCV_LIBS})

# Filter
add_executable(Filter Filter.cpp)
target_link_libraries(Filter ${OpenCV_LIBS})

# Border
add_executable(Border Border.cpp)
target_link_libraries(Border ${OpenCV_LIBS})

# Sobel
add_executable(Sobel Sobel.cpp)
target_link_libraries(Sobel ${OpenCV_LIBS})

# Laplace
add_executable(Laplace Laplace.cpp)
target_link_libraries(Laplace ${OpenCV_LIBS})

# EdgeDetector
add_executable(EdgeDetector EdgeDetector.cpp)
target_link_libraries(EdgeDetector ${OpenCV_LIBS})

# HoughLine
add_executable(HoughLine HoughLine.cpp)
target_link_libraries(HoughLine ${OpenCV_LIBS})

# HoughCircle
add_executable(HoughCircle HoughCircle.cpp)
target_link_libraries(HoughCircle ${OpenCV_LIBS})

# Mapping
add_executable(Mapping Mapping.cpp)
target_link_libraries(Mapping ${OpenCV_LIBS})

# Affine
add_executable(Affine Affine.cpp)
target_link_libraries(Affine ${OpenCV_LIBS})

# EqualizeHist
add_executable(EqualizeHist EqualizeHist.cpp)
target_link_libraries(EqualizeHist ${OpenCV_LIBS})

# Histgram
add_executable(Histgram Histgram.cpp)
target_link_libraries(Histgram ${OpenCV_LIBS})

# HistComparison
add_executable(HistComparison HistComparison.cpp)
target_link_libraries(HistComparison ${OpenCV_LIBS})

# BackProjection
add_executable(BackProjection BackProjection.cpp)
target_link_libraries(BackProjection ${OpenCV_LIBS})

# TemplateMatching
add_executable(TemplateMatching TemplateMatching.cpp)
target_link_libraries(TemplateMatching ${OpenCV_LIBS})

# Contours
add_executable(Contours Contours.cpp)
target_link_libraries(Contours ${OpenCV_LIBS})

# ConvexHull
add_executable(ConvexHull ConvexHull.cpp)
target_link_libraries(ConvexHull ${OpenCV_LIBS})

# BoundingContours
add_executable(BoundingContours BoundingContours.cpp)
target_link_libraries(BoundingContours ${OpenCV_LIBS})

# EllipseContours
add_executable(EllipseContours EllipseContours.cpp)
target_link_libraries(EllipseContours ${OpenCV_LIBS})

# ImageMoments
add_executable(ImageMoments ImageMoments.cpp)
target_link_libraries(ImageMoments ${OpenCV_LIBS})

# PolygonTest
add_executable(PolygonTest PolygonTest.cpp)
target_link_libraries(PolygonTest ${OpenCV_LIBS})
