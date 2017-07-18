cl_program CreateProgram (cl_context context, cl_device_id device, const char * fileName)
{
  cl_int errNum;
  cl_program program;

  ifstream kernelFile (fileName, ios::in);
  if (!kernelFile.is_open())
  {
    cerr << "Failed to open file for reading: " << fileName << endl;
    return NULL;
  }

  ostringstream oss;
  oss << kernelFile.rdbuf();

  string srcStdStr = oss.str();
  const char * srcStr = srcStdStr.c_str();
  program = clCreateProgramWithSource (context, 1, (const char**)&srcStr, NULL, NULL);
  if (program == NULL)
  {
    cerr << "Failed to create CL program from source." << endl;
    return NULL;
  }

  errNum = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    // determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    cerr << "Error in kernel: " << endl;
    cerr << buildLog;
    clReleaseProgram (program);
    return NULL;
  }

  return program;
}
