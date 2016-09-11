#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getrobotvelocity robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get a simplified absolute robot velocity in FRAME_ROBOT.
  std::vector<float> result = motion.getRobotVelocity();

  std::cout << "Robot velocity is: " << result << std::endl;

  return 0;
}
