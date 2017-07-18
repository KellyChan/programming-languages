cl_command_queue CreateCommandQueue (cl_context context, 
                                     cl_device_id * device)
{
  cl_int errNum;
  cl_device_id * devices;
  cl_command_queue commandQueue = NULL;
  size_t deviceBufferSize = -1;

  // first get the size of the devices buffer
  errNum = clGetContextInfo (context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum != CL_SUCCESS)
  {
    cerr << "Failed call to clGetContextInfo (..., CL_CONTEXT_DEVICES, ...)";
    return NULL;
  }

  if (deviceBufferSize < 0)
  {
    cerr << "No devices available";
    return NULL;
  }

  // allocate memory for the devices buffer
  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo (context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

  if (errNum != CL_SUCCESS)
  {
    cerr << "Failed to get device IDs";
    return NULL;
  }

  commandQueue = clCreateCommandQueue (context, devices[0], 0, NULL);
  if (commandQueue == NULL)
  {
    cerr << "Failed to create commandQueue for device 0.";
    return NULL;
  }

  * device = devices[0];
  delete [] devices;
  return commandQueue;
}
