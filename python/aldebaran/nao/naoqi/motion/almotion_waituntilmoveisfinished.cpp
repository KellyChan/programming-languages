#include <iostream>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>
#include <qi/os.hpp>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_waituntilmoveisfinished robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);
  AL::ALRobotPostureProxy robotPosture(robotIp);

  robotPosture.goToPosture("StandInit", 0.5f);

  // Example showing how to use waitUntilMoveIsFinished.
  // Start a walk
  float x = 0.10f;
  float y = 0.0f;
  float theta = 0.0f;
  motion.post.moveTo(x, y, theta);

  // Wait for it to finish
  motion.waitUntilMoveIsFinished();
  // Then do something else

  return 0;
}
