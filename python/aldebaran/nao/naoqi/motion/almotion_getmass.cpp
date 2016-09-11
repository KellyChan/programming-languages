#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getmass robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the mass of "HeadYaw".
  std::string pName = "HeadYaw";
  float mass = motion.getMass(pName);

  std::cout << "Mass on " << pName << " is " << mass << std::endl;

  return 0;
}
