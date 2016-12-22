########################################
CUBLAS
########################################


::

    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime.h>
    #include "cublas_v2.h"

    # define n 6

    int main(void)
    {
        cudaError_t cudaStat;  // cudaMalloc status
        cublasStatus_t stat;  // CUBLAS function status
        cublasHandle_t handle; // CUBLAS context

        // ON THE HOST
        // create variables
        // ...
        // x = (float*)malloc(n*sizeof(*x));
    
        // ON THE DEVICE
        float* d_x;
        cudaStat = cudaMalloc((void**)& d_x, n*sizeof(*x));
        stat = cublasCreate(&handle);
        stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
        

        // Level 1 - scalar and vector based ops
        int result;
        stat = cublasIsamin(handle, n, d_x, 1, &result);
        stat = cublasSasum(handle, n, d_x, 1, &result);
        stat = cublasSaxpy(handle, n, &a1, d_x, 1, d_y, 1);  // y = ax + y
        stat = cublasScopy(handle, n, d_x, 1, d_y, 1);  // copy vector x to vector y
        stat = cublasSdot(handle, n, d_x, 1, d_y, 1, &result);  // x*y = x0y0 + ... + x_(n-1)y_(n-1)
        stat = cublasSnrm2(handle, n, d_x, 1, &result);
        stat = cublasSrot(handle, n, d_x, 1, d_y, 1, &c, &s);  // apply the given rotation
        stat = cublasSrotg(handle, &a, &b, &c, &s);  // construct the given rotation matrix
        stat = cublasSrotm(handle, n, d_x, 1, d_y, 1, param);  // apply the modified given rotation
        stat = cublasSrotmg(handle, &d1, &d2, &x1, &y1, param);  
        stat = cublasSscal(handle, n, &a1, d_x, 1);  // x = ax
        stat = cublasSswap(handle, n, d_x, 1, d_y, 1);  // x <-> y
        
        stat = cublasGetVector(n, sizeof(float), d_y, 1, y, 1);


        // Level 2 - Matrix-vector ops
        stat = cublasSgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, &1, d_a, m d_x, 1, &bet, d_y, 1);
        stat = cublasSgemv(handle, CUBLAS_OP_N, m, n, &a1, d_a, m, d_x, 1, &bet, d_y, 1);  // y = a op(A)x + by
        stat = cublasSger(handle, m, n, &a1, d_x, 1, d_y, 1, d_a, m); // rank one update 

        // free the memory
        cudaFree(d_x);
        cublasDestroy(handle);
        free(x);

        return EXIT_SUCCESS;
        
    }
