#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_wbfootstate robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  std::vector<float> stiffnesses = motion.getStiffnesses("Body");
  if (stiffnesses[0] < 0.5f) {
    std::cerr << "Warning: this example will have no effect. "
              << "Robot must be stiff and standing." << std::endl;
    return 1;
  }

  bool isEnabled = true;
  motion.wbEnable(isEnabled);

  // Example showing how to fix the feet.
  std::string stateName = "Fixed";
  std::string supportLeg = "Legs";
  motion.wbFootState(stateName, supportLeg);
  std::cout << supportLeg << " " << stateName << std::endl;

  // Example showing how to fix the left leg and constrained in a plane the right leg.
  motion.wbFootState("Fixed", "LLeg");
  motion.wbFootState("Plane", "RLeg");
  std::cout << "LLeg Fixed, RLeg Plane" << std::endl;

  // Example showing how to fix the left leg and keep free the right leg.
  motion.wbFootState("Fixed", "LLeg");
  motion.wbFootState("Free", "RLeg");
  std::cout << "LLeg Fixed, RLeg Free" << std::endl;

  isEnabled = false;
  motion.wbEnable(isEnabled);

  return 0;
}
