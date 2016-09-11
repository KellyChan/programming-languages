#include <iostream>
#include <alproxies/almotionproxy.h>
#include <qi/os.hpp>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_changeangles robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Setting head stiffness on.
  motion.setStiffnesses("Head", 1.0f);

  // Example showing a slow, relative move of "HeadYaw".
  // Calling this multiple times will move the head further.
  AL::ALValue names      = "HeadYaw";
  AL::ALValue changes    = 0.25f;
  float fractionMaxSpeed = 0.05f;
  motion.changeAngles(names, changes, fractionMaxSpeed);

  qi::os::sleep(1.0f);

  // Setting head stiffness off.
  motion.setStiffnesses("Head", 0.0f);

  return 0;
}
