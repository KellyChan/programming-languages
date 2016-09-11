#include <qi/os.hpp>
#include <iostream>
#include <alproxies/almotionproxy.h>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_killtasksusingresources robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Example showing how to killTasksUsingResources
  // We will create a task first, so that we see a result
  AL::ALValue names      = "HeadYaw";
  AL::ALValue angleLists = 1.0f;
  AL::ALValue timeList   = 3.0f;
  bool isAbsolute        = true;
  motion.setStiffnesses(names, 1.0f);
  motion.post.angleInterpolation(names, angleLists, timeList, isAbsolute);
  qi::os::msleep(1000);
  std::cout << "Killing task..." << std::endl;
  motion.killTasksUsingResources(AL::ALValue::array(names));
  std::cout << "Task killed." << std::endl;

  motion.setStiffnesses(names, 0.0f);

  return 0;
}
