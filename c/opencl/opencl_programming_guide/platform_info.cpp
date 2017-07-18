void displayInfo (void)
{
  cl_int errNum;
  cl_uint numPlatform;
  cl_platform_id * platformIds;
  cl_context context = NULL;

  // query the total number of platforms
  errNum = clGetPlatformIDs (0, NULL, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms <= 0)
  {
    std::cerr << "Failed to find any OpenCL platform." << std::endl;
    return;
  }

  // allocate memory for the installed platform, and query to get the list
  platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

  // query the total number of platforms
  errNum = clGetPlatformIDs (numPlatforms, platformIds, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    return;
  }

  std::cout << "Number of platforms: \t" << numPlatforms << std::endl;

  // iterate through the list of platforms displaying associated info
  for (cl_uint i = 0; i < numPlatforms; i++)
  {
    // display information associated with the platform
    DisplayPlatformInfo (platformIds[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
    DisplayPlatformInfo (platformIds[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
    DisplayPlatformInfo (platformIds[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
    DisplayPlatformInfo (platformIds[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
  }
}


void DisplayPlatformInfo (cl_platform_id id, cl_platform_info name, std::string str)
{
  cl_int errNum;
  std::size_t paramValueSize;

  errNum = clGetPlatformInfo (id, name, 0, NULL, &paramValueSize);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Failed to find OpenCL platform" << str << "." << std::endl;
    return;
  }

  char * info = (char *)alloca(sizeof(char) * paramValueSize);
  errNum = clGetPlatformInfo (id, name, paramValueSize, info, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
    return;
  }

  std::cout << "\t" << str << ":\t" << info << std::endl;
}
