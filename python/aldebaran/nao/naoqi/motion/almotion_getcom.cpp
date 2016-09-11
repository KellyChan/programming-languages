#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getcom robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the COM of "HeadYaw".
  std::string pName = "HeadYaw";
  int pSpace = 0; // FRAME_TORSO
  bool pUseSensors = false;
  std::vector<float> pos = motion.getCOM(pName, pSpace, pUseSensors);

  std::cout << pName << " COM position: " << std::endl;
  std::cout << "x: " << pos[0] << "; y: " << pos[1] << "; z: " << pos[2]
            << std::endl;

  return 0;
}
