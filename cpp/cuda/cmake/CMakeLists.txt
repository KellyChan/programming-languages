CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#--------------------------------------------
PROJECT(hellocuda)


FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)

INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(/usr/local/cuda/include ${MPI_INCLUDE_PATH})

FILE(GLOB SOURCES 
    "src/*.cu" "src/*.cpp" "src/*.c" "src/*.h"
    "src/utils/*.cu" "src/utils/*.cpp" "src/utils/*.c" "src/utils/*.h"
)

#--------------------------------------------

CUDA_ADD_EXECUTABLE(hellocuda ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -03 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20, code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30, code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35, code=sm_35)

TARGET_LINK_LIBRARIES(hellocuda /usr/local/cuda/lib64/libcudart.so ${MPI_LIBRARIES})

