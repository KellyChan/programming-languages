#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_setBreathConfig robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to change the breathing configuration
  // Setting a relaxed configuration: 5 breaths per minute at max amplitude
  AL::ALValue breathConfig;
  breathConfig.arraySetSize(2);

  AL::ALValue tmp;
  tmp.arraySetSize(2);
  tmp[0] = "Bpm";
  tmp[1] = 5.0f;
  breathConfig[0] = tmp;

  tmp[0] = "Amplitude";
  tmp[1] = 1.0f;
  breathConfig[1] = tmp;
  motion.setBreathConfig(breathConfig);

  return 0;
}
