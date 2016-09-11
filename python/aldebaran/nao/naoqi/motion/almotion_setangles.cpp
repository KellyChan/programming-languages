#include <iostream>
#include <alproxies/almotionproxy.h>
#include <qi/os.hpp>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_setangles robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to set angles, using a fraction of max speed
  AL::ALValue names       = AL::ALValue::array("HeadYaw", "HeadPitch");
  AL::ALValue angles      = AL::ALValue::array(0.3f, -0.3f);
  float fractionMaxSpeed  = 0.1f;
  motion.setStiffnesses(names, AL::ALValue::array(1.0f, 1.0f));
  qi::os::sleep(1.0f);
  motion.setAngles(names, angles, fractionMaxSpeed);
  qi::os::sleep(1.0f);
  motion.setStiffnesses(names, AL::ALValue::array(0.0f, 0.0f));

  return 0;
}
