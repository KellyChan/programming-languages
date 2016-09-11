#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getposition robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the position of the top camera
  std::string name = "CameraTop";
  int space = 1;
  bool useSensorValues = true;
  std::vector<float> result = motion.getPosition(name, space, useSensorValues);
  std::cout << name << ":" << std::endl;
  std::cout << "Position (x, y, z): " << result.at(0) << ", " << result.at(1)
            << ", " << result.at(2) << std::endl;
  std::cout << "Rotation (x, y, z): " << result.at(3) << ", " << result.at(4)
            << ", " << result.at(5) << std::endl;

  return 0;
}
