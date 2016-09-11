#include <iostream>
#include <alproxies/alledsproxy.h>

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage: leds_on pIp" << std::endl;
    return 1;
  }
  const std::string pIp = argv[1];

  // Create a proxy to ALLeds.
  AL::ALLedsProxy leds(pIp);

  // Example showing how to switch on a group
  std::string name = "FaceLeds";
  leds.on(name);

  std::cout << "LEDs from " << name << " group are now on." << std::endl;

  return 0;
}
