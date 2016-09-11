#include <iostream>
#include <alproxies/alledsproxy.h>

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage: leds_ledsfade pIp" << std::endl;
    return 1;
  }
  const std::string pIp = argv[1];

  // Create a proxy to ALLeds.
  AL::ALLedsProxy leds(pIp);

  // Example showing how to fade the left eye group to mid-intensity
  std::string name = "EarLeds";
  float intensity = 0.5f;
  float duration = 1.0f;
  leds.fade(name, intensity, duration);
  std::cout << "Faded " << name << " to " << intensity << " in " << duration
            << std::endl;

  return 0;
}
