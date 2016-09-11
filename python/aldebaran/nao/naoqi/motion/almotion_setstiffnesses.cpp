#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_setstiffnesses robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to set the stiffness to 1.0.
  // Beware, doing this could be dangerous, it is safer to use the
  // stiffnessInterpolation method which takes a duration parameter.
  std::string names  = "Body";
  // If only one parameter is received, this is applied to all joints
  float stiffnesses  = 1.0f;
  motion.setStiffnesses(names, stiffnesses);

  return 0;
}
