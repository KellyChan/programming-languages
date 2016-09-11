#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getMoveArmsEnable robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the enabled flags for the arms
  std::cout << "LeftArmEnabled: " << motion.getMoveArmsEnabled("LArm") << std::endl;
  std::cout << "RightArmEnabled: " << motion.getMoveArmsEnabled("RArm") << std::endl;
  std::cout << "BothArmsEnabled: " << motion.getMoveArmsEnabled("Arms") << std::endl;
  return 0;
}
