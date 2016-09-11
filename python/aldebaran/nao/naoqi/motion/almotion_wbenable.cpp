#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_wbenable robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to active Whole Body Balancer.
  bool isEnabled = true;
  motion.wbEnable(isEnabled);
  std::cout << "Whole body enabled." << std::endl;

  isEnabled = false;
  motion.wbEnable(isEnabled);
  std::cout << "Whole body disabled." << std::endl;

  return 0;
}
