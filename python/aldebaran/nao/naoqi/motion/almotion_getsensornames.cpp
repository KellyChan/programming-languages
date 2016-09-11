#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getsensornames robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Gets the list of sensors supported on your robot.
  std::vector<std::string> sensorList = motion.getSensorNames();
  std::cout << "Sensors: " << std::endl;
  for (unsigned int i=0; i<sensorList.size(); i++)
  {
     std::cout << sensorList.at(i) << std::endl;
  }

  return 0;
}
