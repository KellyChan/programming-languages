#include <stdio.h>
#include <stdlib.h>
  
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define NUM_ELEMENTS 16
 
/* A kernel which sets all elements of an array to -1 */
const char * source_str  = "__kernel void setminusone(__global int *out)"
                           "{"
                           "    int i = get_global_id(0);"
                           "    out[i] = -1;"
                           "}";
 
int main(void) {
 
    /* Get platform and device information */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;  
    cl_uint num_devices;
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
  
    /* Create an OpenCL context */
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    /* Create a command queue */
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    /* Allocate host array */
    int *host_arr = malloc(NUM_ELEMENTS * sizeof(*host_arr));
 
    /* Create device memory buffer */
    cl_mem dev_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        NUM_ELEMENTS * sizeof(*host_arr), NULL, &ret);
    /* Create a program from the kernel source */
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, NULL, &ret);
  
    /* Build the program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  
    /* Create the OpenCL kernel */
    cl_kernel kernel = clCreateKernel(program, "setminusone", &ret);
 
    /* Set the kernel argument */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_mem_obj);
 
    /* Execute the OpenCL kernel */
    size_t global_item_size = NUM_ELEMENTS;
    size_t local_item_size = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_item_size, &local_item_size, 0, NULL, NULL);
 
    /* Read the results from the device memory buffer back into host array */
    ret = clEnqueueReadBuffer(command_queue, dev_mem_obj, CL_TRUE, 0,
                              NUM_ELEMENTS * sizeof(*host_arr), host_arr, 0, NULL, NULL);
 
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
 
    if (ret != CL_SUCCESS) {
        printf("OpenCL error executing kernel: %d\n", ret);
        goto cleanup;
    }
 
    /* Print out the result element by element */
    size_t i;
    for (i = 0; i < NUM_ELEMENTS; ++i)
        printf("%d ", host_arr[i]);
    printf("\n");
 
cleanup:
    /* Clean up */
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(dev_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}
