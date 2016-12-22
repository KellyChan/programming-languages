CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#--------------------------------------------
PROJECT(hellocuda)


FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)

INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} ${MPI_INCLUDE_PATH} ${GLEW_INCLUDE_PATH})


FILE(GLOB SOURCES 
    "src/*.cu" "src/*.cpp" "src/*.c" "src/*.h"
    "src/utils/*.cu" "src/utils/*.cpp" "src/utils/*.c" "src/utils/*.h"
)

#--------------------------------------------

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
# LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O2 -ffast-math -Wall")
# LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
# LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
# LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
# LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
CUDA_ADD_EXECUTABLE (hellocuda ${SOURCES})
# TARGET_LINK_LIBRARIES(${CUDA_LIBRARIES} ${MPI_LIBRARIES} ${OPENGL_LIBRARIES})

