#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getlimits robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the limits for the whole body
  std::string name = "Body";
  std::vector<std::string> jointNames = motion.getBodyNames(name);
  AL::ALValue limits = motion.getLimits(name);
  for (unsigned int i=0; i<limits.getSize(); i++)
  {
    std::cout << " Joint name " << jointNames[i]
              << " minAngle " << limits[i][0] << " rad"
              << " maxAngle " << limits[i][1] << " rad"
              << " maxChange " << limits[i][2] << " rad.s-1"
              << " maxTorque " << limits[i][3] << " N.m"
              << std::endl;
  }

  return 0;
}
