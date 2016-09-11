#include <alproxies/alvideodeviceproxy.h>

#include <iostream>

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage: videodevice_getactivecamera pIp" << std::endl;
    return 1;
  }
  const std::string pIp = argv[1];

  // Proxy to ALVideoDevice.
  AL::ALVideoDeviceProxy cameraProxy(pIp);

  int activeCam = cameraProxy.getActiveCamera();

  std::cout << "Active camera is " << activeCam << std::endl;
  return 0;
}
