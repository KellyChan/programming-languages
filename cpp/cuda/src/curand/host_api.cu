/*
    This program uses the host CURAND API to generate 100 pseudorandom floats.
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>


#define CUDA_CALL(x) do {
    if ( (x) != cudaSuccess )
    {
        printf("Error at %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
} while (0)

#define CURAND_CALL(x) do {
    if ( (x) != CURAND_STATUS_SUCCESS )
    {
        printf("Error at %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
} while (0)
