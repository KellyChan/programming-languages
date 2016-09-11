#include <iostream>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_wbgotobalance robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);
  AL::ALRobotPostureProxy robotPosture(robotIp);

  robotPosture.goToPosture("StandInit", 0.5f);

  bool isEnabled = true;
  motion.wbEnable(isEnabled);

  // Example showing how to com go to LLeg.
  std::string supportLeg = "LLeg";
  float duration         = 2.0f;
  motion.wbGoToBalance(supportLeg, duration);

  isEnabled = false;
  motion.wbEnable(isEnabled);

  return 0;
}
