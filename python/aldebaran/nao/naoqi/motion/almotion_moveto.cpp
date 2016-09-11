#include <iostream>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_moveto robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);
  AL::ALRobotPostureProxy robotPosture(robotIp);

  robotPosture.goToPosture("StandInit", 0.5f);

  // Example showing the moveTo command
  // as length of path is less than 0.4m
  // the path will be an SE3 interpolation
  // The units for this command are meters and radians
  float x  = 0.2f;
  float y  = 0.2f;
  // pi/2 anti-clockwise (90 degrees)
  float theta = 1.5709f;
  motion.moveTo(x, y, theta);
  // Will block until walk Task is finished

  // Example showing the moveTo command
  // as length of path is more than 0.4m
  // the path will be follow a dubins curve
  // The units for this command are meters and radians
  x  = -0.1f;
  y  = -0.5f;
  theta  = 0.0f;
  motion.moveTo(x, y, theta);

  return 0;
}
