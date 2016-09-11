#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_iscollision robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to get the collision state
  std::string pChainName = "LArm";
  std::string collisionState = motion.isCollision(pChainName);

  std::cout << pChainName << " collision state: " << collisionState << std::endl;
  return 0;
}
