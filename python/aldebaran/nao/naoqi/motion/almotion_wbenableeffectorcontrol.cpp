#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_wbenableeffectorcontrol robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to active Head tracking.
  std::string effectorName = "Head";
  bool isEnabled = true;
  motion.wbEnableEffectorControl(effectorName, isEnabled);
  std::cout << "Enabled whole body effector " << effectorName << " control"
            << std::endl;

  isEnabled = false;
  motion.wbEnableEffectorControl(effectorName, isEnabled);
  std::cout << "Disabled whole body effector " << effectorName << " control"
            << std::endl;

  return 0;
}
