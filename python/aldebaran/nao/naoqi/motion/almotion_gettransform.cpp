#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_gettransform robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the end of the right arm as a transform
  // represented in torso space.
  std::string name  = "RArm";
  int space  = 0; // TORSO_SPACE
  bool useSensorValues  = false;
  std::vector<float> result = motion.getTransform(name, space, useSensorValues);
  // R R R x
  // R R R y
  // R R R z
  // 0 0 0 1
  std::cout << "Transform of RArm" << std::endl;
  std::cout << result.at(0) << " " << result.at(1) << " " << result.at(2) << " " << result.at(3) << std::endl;
  std::cout << result.at(4) << " " << result.at(5) << " " << result.at(6) << " " << result.at(7) << std::endl;
  std::cout << result.at(8) << " " << result.at(9) << " " << result.at(10) << " " << result.at(11) << std::endl;

  return 0;
}
