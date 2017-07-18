cl_context CreateContext ()
{
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_context context = NULL;

  // select an opencl platform to run on
  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCESS || numPlatforms <= 0)
  {
    cerr << "Failed to find any OpenCL platforms." << endl;
    return NULL;
  }

  // create an opencl context on the platforom
  // attempt to create a GPU-based context, and if that fails,
  // try to create a CPU-based context.
  cl_context_properties contextProperties[] =
  {
    CL_CONTEXT_PLATFORM, 
    (cl_context_properties)firstPlatformId,
    0
  };

  context = clCreateContextFromType (contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
  if (errNum != CL_SUCESS)
  {
    cout << "Could not create GPU context, trying CPU..." << endl;
    context = clCreateContextFromType (contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCESS)
    {
      cerr << "Failed to create an OpenCL GPU or CPU context.";
      return NULL;
    }
  }

  return context;
}
