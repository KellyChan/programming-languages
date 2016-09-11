#include <iostream>
#include <alproxies/almotionproxy.h>
#include <qi/os.hpp>

int main(int argc, char **argv)
{
  std::string robotIp = "127.0.0.1";

  if (argc < 2) {
    std::cerr << "Usage: almotion_angleinterpolationBezier robotIp "
              << "(optional default \"127.0.0.1\")."<< std::endl;
  }
  else {
    robotIp = argv[1];
  }

  AL::ALMotionProxy motion(robotIp);

  // Setting head stiffness on.
  motion.setStiffnesses("Head", 1.0f);

  // Example showing a single target angle for one joint
  // Interpolates the head yaw to 1.0 radian in 1.0 second
  std::vector<std::string> names;
  AL::ALValue timeLists;
  AL::ALValue angleLists;

  names.push_back("HeadYaw");
  timeLists.arrayPush(AL::ALValue::array(1.0f));
  angleLists.arrayPush(AL::ALValue::array(1.0f));

  std::cout << "Step 1: single target angle for one joint" << std::endl;
  motion.angleInterpolationBezier(names, timeLists, angleLists);

  qi::os::sleep(1.0f);

  // Example showing a single trajectory for one joint
  // Interpolates the head yaw to 1.0 radian and back to zero in 2.0 seconds
  timeLists.clear();
  timeLists.arrayPush(AL::ALValue::array(1.0f, 2.0f));

  angleLists.clear();
  angleLists.arrayPush(AL::ALValue::array(1.0f, 0.0f));
  std::cout << "Step 2: single trajectory for one joint" << std::endl;
  motion.angleInterpolationBezier(names, timeLists, angleLists);

  qi::os::sleep(1.0f);

  // Example showing multiple trajectories
  // Interpolates the HeadYaw to 1.0 radian and back to zero in 2.0 seconds
  // while interpolating HeadPitch up and down over a longer period.
  names.at(0) = "Head";
  // Each joint can have lists of different lengths, but the number of
  // angles and the number of times must be the same for each joint.
  // Here, the second joint ("HeadPitch") has three angles, and
  // three corresponding times.
  timeLists.clear();
  timeLists.arraySetSize(2);
  timeLists[0] = AL::ALValue::array(1.0f, 2.0f);
  timeLists[1] = AL::ALValue::array(1.0f, 2.0f, 3.0f);

  angleLists.clear();
  angleLists.arraySetSize(2);
  angleLists[0] = AL::ALValue::array(1.0f, 0.0f);
  angleLists[1] = AL::ALValue::array(-0.5f, 0.5f, 0.0f);
  std::cout << "Step 3: multiple trajectories" << std::endl;
  motion.angleInterpolationBezier(names, timeLists, angleLists);

  // Setting head stiffness off.
  motion.setStiffnesses("Head", 0.0f);

  return 0;
}
