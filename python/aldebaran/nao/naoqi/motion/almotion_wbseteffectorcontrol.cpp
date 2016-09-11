#include <iostream>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_wbseteffectorcontrol robotIp "
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

  // Example showing how to set orientation target for Head tracking.
  std::string effectorName = "Head";
  AL::ALValue targetCoordinate = AL::ALValue::array(0.1f, 0.0f, 0.0f);
  motion.wbSetEffectorControl(effectorName, targetCoordinate);

  isEnabled = false;
  motion.wbEnable(isEnabled);

  return 0;
}
