#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getbodynames robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the names of all the joints in the body.
  std::vector<std::string> bodyNames = motion.getBodyNames("Body");
  std::cout << "All joints in Body: " << std::endl;
  std::cout << bodyNames << std::endl;

  // Example showing how to get the names of all the joints in the left leg.
  std::vector<std::string> leftLegJointNames = motion.getBodyNames("LLeg");
  std::cout << "All joints in LLeg: " << std::endl;
  std::cout << leftLegJointNames << std::endl;

  return 0;
}
