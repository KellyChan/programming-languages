#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getstiffnesses robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the Body stiffnesses
  std::string jointName = "Body";
  std::vector<float> stiffnesses = motion.getStiffnesses(jointName);

  std::cout << jointName << "stiffnesses: " << stiffnesses << std::endl;

  return 0;
}
