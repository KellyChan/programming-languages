#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_wbenablebalanceconstraint robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to Constraint Balance Motion.
  bool isEnable = true;
  std::string supportLeg  = "Legs";
  motion.wbEnableBalanceConstraint(isEnable, supportLeg);
  std::cout << "Enabled whole body balance constraint on " << supportLeg << std::endl;

  isEnable = false;
  motion.wbEnableBalanceConstraint(isEnable, supportLeg);
  std::cout << "Disabled whole body balance constraint on " << supportLeg << std::endl;

  return 0;
}
