#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage: motion_wbenableeffectoroptimization pIp"
              << std::endl;
    return 1;
  }
  const std::string pIp = argv[1];

  AL::ALMotionProxy motion(pIp);

  // Example showing how to Enable Effector Control as an Optimization.
  std::string effectorName = "RArm";
  bool isActive = true;
  motion.wbEnableEffectorOptimization(effectorName, isActive);
  std::cout << "Enabled whole body effector " << effectorName << " optimization"
          << std::endl;

  isActive = false;
  motion.wbEnableEffectorOptimization(effectorName, isActive);
  std::cout << "Disabled whole body effector " << effectorName << " optimization"
          << std::endl;

  return 0;
}
