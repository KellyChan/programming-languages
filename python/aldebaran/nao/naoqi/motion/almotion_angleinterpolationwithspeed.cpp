#include <iostream>
#include <alproxies/almotionproxy.h>
#include <qi/os.hpp>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_angleinterpolationwithspeed robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Setting head stiffness on.
  motion.setStiffnesses("Head", 1.0f);

  // Example showing a single target for one joint
  AL::ALValue names        = "HeadYaw";
  AL::ALValue targetAngles = 1.0f;
  float maxSpeedFraction   = 0.2f; // Using 20% of maximum joint speed
  std::cout << "Step 1: single target for one joint." << std::endl;
  motion.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction);

  qi::os::sleep(1.0f);

  // Example showing multiple joints
  // Instead of listing each joint, you can use a chain name, which will
  // be expanded to contain all the joints in the chain. In this case,
  // "Head" will be interpreted as ["HeadYaw", "HeadPitch"]
  names  = "Head";
  // We still need to specify the correct number of target angles
  targetAngles = AL::ALValue::array(0.5f, 0.25f);
  maxSpeedFraction  = 0.2f; // Using 20% of maximum joint speed
  std::cout << "Step 2: multiple joints." << std::endl;
  motion.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction);

  // Setting head stiffness off.
  motion.setStiffnesses("Head", 0.0f);

  return 0;
}
