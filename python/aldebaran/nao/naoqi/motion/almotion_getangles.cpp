#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_getangles robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example that finds the difference between the command and sensed angles.
  std::string names = "Body";
  bool useSensors   = false;
  std::vector<float> commandAngles = motion.getAngles(names, useSensors);
  std::cout << "Command angles: " << std::endl << commandAngles << std::endl;
  useSensors = true;
  std::vector<float> sensorAngles = motion.getAngles(names, useSensors);
  std::cout << "Sensor angles: " << std::endl << sensorAngles << std::endl;

  return 0;
}
