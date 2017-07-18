bool saveProgramBinary (cl_program program, cl_device_id device, const char * fileName)
{
  cl_uint numDevices = 0;
  cl_int = errNum;

  // 1 - query for number of devices attached to program
  errNum = clGetProgramInfo (program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error querying for number of devices." << std::endl;
    return false;
  }

  // 2 - get all of the device ids
  cl_device_id  * devices = new cl_device_id (numDevices);
  errNum = clGetProgramInfo (program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error querying for devices." << std::endl;
    delete [] devices;
    return false;
  }

  // 3 - determine the size of each program binary
  size_t * programBinarySizes = new size_t (numDevices);
  errNum = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES, 
                             sizeof(size_t) * numDevices,
                             programBinarySizes, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error querying for program binary sizes." << std::endl;
    delete [] devices;
    delete [] programBinarySizes;
    return false;
  }

  unsigned char ** programBinaries = new unsigned char * [numDevices];
  for (cl_uint i = 0; i < numDevices; ++i)
  {
    programBinaries[i] = new unsigned char[programBinarySizes[i]];
  }

  // 4 - get all of the program binaries
  errNum = clGetProgramInfo (program, CL_PROGRAM_BINARIES,
                             sizeof(unsigned char*) * numDevices,
                             programBinaries, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error querying for program binaries" << std::endl;
    delete [] devices;
    delete [] programBinarySizes;
    for (cl_uint i = 0; i < numDevices; ++i)
    {
      delete [] programBinaries[i];
    }
    delete [] programBinaries;
    return false;
  }

  // 5 - finally store the binaries for the device requested out to disk for future reading
  for (cl_uint i = 0; i < numDevices; ++i)
  {
    if (devices[i] == device)
    {
      FILE * fp = fopen(fileName, "wb");
      fwrite (programBinaries[i], 1, programBinarySizes[i], fp);
      fclose(fp);
      break;
    }
  }

  // clean up
  delete [] devices;
  detele [] programBinarySizes;
  for (cl_uint i = 0; i < numDevices; ++i)
  {
    delete [] programBinaries[i];
  }
  delete [] programBinaries;
  return true;
}


cl_program CreateProgramFromBinary (cl_context context,
                                    cl_device_id device,
                                    const char * fileName)
{
  FILE * fp = fopen(fileName, "rb");
  if (fp == NULL)
  {
    return NULL;
  }

  // determine the size of the binary
  size_t binarySize;
  fseek (fp, 0, SEEK_END);
  binarySize = ftell (fp);
  rewind (fp);

  // load binary from disk
  unsigned char * programBinary = new unsigned char[binarySize];
  fread (programBinary, 1, binarySize, fp);
  fclose (fp);

  cl_int errNum = 0;
  cl_program program;
  cl_int binaryStatus;

  program = clCreateProgramWithBinary (context, 1, &device,
                                       &binarySize, 
                                       (const unsigned char**)&programBinary,
                                       &binaryStatus,
                                       &errNum);
  delete [] programBinary;
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error loading program binary." << std::endl;
    return NULL;
  }

  if (binaryStatus != CL_SUCCESS)
  {
    std::cerr << "Invalid binary for device" << std::endl;
    return NULL;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    // determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    std::cerr << "Error in program: " << std::endl;
    clReleaseProgram (program);
    return NULL;
  }
  return program;
}
