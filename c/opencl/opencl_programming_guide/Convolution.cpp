#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

// constants
const unsigned int inputSignalWidth = 8;
const unsigned int inputSignalHeight = 8;

cl_uint inputSignal[inputSignalWidth][inputSignalHeight] =
{
  {3, 1, 1, 4, 8, 2, 1, 3},
  {4, 2, 1, 1, 2, 1, 2, 3},
  {4, 4, 4, 4, 3, 2, 2, 2},
  {9, 8, 3, 8, 9, 0, 0, 0},
  {9, 3, 3, 9, 0, 0, 0, 0},
  {0, 9, 0, 8, 0, 0, 0, 0},
  {3, 0, 8, 8, 9, 4, 4, 4},
  {5, 9, 8, 1, 8, 1, 1, 1}
};


const unsigned int outputSignalWidth = 6;
const unsigned int outputSignalHeigth = 6;

cl_uint outputSignal[outputSignalWidth][outputSignalHeight];

const unsigned int maskWidth = 3;
const unsigned int maskHeight = 3;

cl_uint mask[maskWidth][maskHeight] = 
{
  {1, 1, 1},
  {1, 0, 1},
  {1, 1, 1},
};


inline void checkErr (cl_int err, const char * name)
{
  if (err != CL_SUCCESS)
  {
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit (EXIT_FAILURE);
  }
}


void CL_CALLBACK contextCallback (const char * errInfo,
                                  const void * private_info,
                                  size_t cb,
                                  void * user_data)
{
  std::cout << "Error occurred during context use: " << errInfo << std::endl;
  exit (EXIT_FAILURE);
}


int main (int argc, char ** argv)
{
  cl_int errNum;
  cl_uint numPlatforms;
  cl_uint numDevices;
  cl_paltform_id * platformIDs;
  cl_device_id * deviceIDs;
  cl_context context = NULL;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem inputSignalBuffer;
  cl_mem outputSignalBuffer;
  cl_mem maskBuffer;

  errNum = clGetPlatformIDs (0, NULL, &numPlatforms);
  checkErr ((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");
 
  platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
  errNum = clGetPlatformIDs (numPlatforms, platformIDs, NULL);
  checkErr ((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs"); 

  deviceIDs = NULL;
  cl_uint i;
  for (i = 0; i < numPlatforms; i++)
  {
    errNum = clGetDeviceIDs (platformIDs[i], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
      checkErr (errNum, "clGetDeviceIDs");
    }
    else if (numDevices > 0)
    {
      deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
      errNum = clGetDeviceIDs (platformIDs[i], CL_DEVICE_TYPE_CPU, numDevices, &deviceIDs[0], NULL);
      checkErr (errNum, "clGetDeviceIDs");
      break;
    }
  }

  if (deviceIDs == NULL)
  {
    std::cout << "No CPU device found" << std::endl;
    exit (-1);
  }

  cl_context_properties contextProperties[] = 
  {
    CL_CONTEXT_PLATFORM, 
    (cl_context_properties)platformIDs[i], 
    0
  };

  context = clCreateContext (contextProperties, numDevices, deviceIDs, &contextCallback, NULL, &errNum);
  checkErr (errNum, "clCreateContext");

  std::ifstream srcFile ("Convolution.cl");
  checkErr (srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");
  
  std::string srcProg(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

  const char * src = srcProg.c_str();
  size_t length = srcProg.length();

  program = clCreateProgramWithSource (context, 1, &src, &length, &errNum);
  checkErr (errNum, "clCreateProgramWithSource");

  inputSignalBuffer = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
                                      static_cast<void *>(inputSignal), &errNum);
  checkErr (errNum, "clCreateBuffer (inputSignal)");

  maskBuffer = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(cl_uint) * maskHeight * maskWidth,
                               static_cast<void *>(mask), &errNum);
  checkErr (errNum, "clCreateBuffer(mask)");

  outputSignalBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
                                      NULL, &errNum);
  checkErr (errNum, "clCreateBuffer(outputSignal)");

  queue = clCreateCommandQueue (context, deviceIDs[0], 0, &errNum);
  checkErr (errNum, "clCreateCommandQueue");

  errNum = clSetKernelArg (kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
  errNum |= clSetKernelArg (kernel, 1, sizeof(cl_mem), &maskBuffer);
  errNum |= clSetKernelArg (kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
  errNum |= clSetKernelArg (kernel, 3, sizeof(cl_uint), &inputSignalWidth);
  errNum |= clSetKernelArg (kernel, 4, sizeof(cl_uint), &maskWidth);
  checkErr (errNum, "clSetKernelArg");

  const size_t globalWorkSize[1] = { outputSignalWidth * outputSingalHeight };
  const size_t localWorkSize[1] = { 1 };

  errNum = clEnqueueNDRangeKernel (
    queue,
    kernel,
    1,
    NULL,
    globalWorkSize,
    localWorkSize,
    0,
    NULL,
    NULL
  );
  checkErr (errNum, "clEnqueueNDRangeKernel");

  errNum = clEnqueueReadBuffer (queue, outputSignalBuffer, CL_TRUE, 0,
                                sizeof(cl_uint) * outputSignalHeight * outputSignalHeight,
                                outputSignal, 0, NULL, NULL);
  checkErr (errNum, "clEnqueueReadBuffer");

  for (int y = 0; y < outputSignalHeight; y++)
  {
    for (int x = 0; x < outputSignalWidth; x++)
    {
      std::cout << outputSignal[x][y] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
