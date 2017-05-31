int main(int argc, char** argv)
{
 cl_context context = 0;
 cl_command_queue commandQueue = 0;
 cl_program program = 0;
 cl_device_id device = 0;
 cl_kernel kernel = 0;
 cl_mem memObjects[3] = { 0, 0, 0 };
 cl_int errNum;
 // Create an OpenCL context on first available platform
 context = CreateContext();

