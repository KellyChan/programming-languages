#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getrobotconfig robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the robot config
  AL::ALValue robotConfig = motion.getRobotConfig();
  for (unsigned int i=0; i<robotConfig[0].getSize(); i++)
  {
    std::cout << robotConfig[0][i] << ": " << robotConfig[1][i] << std::endl;
  }

  return 0;
}
