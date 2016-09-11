#include <iostream>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_moveinit robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);
  AL::ALRobotPostureProxy robotPosture(robotIp);
  robotPosture.goToPosture("Stand", 0.5f);

  // Example showing how Initialize move process.
  motion.moveInit();

  return 0;
}
