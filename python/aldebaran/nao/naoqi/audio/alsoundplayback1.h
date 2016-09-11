/**
* @author Vincent Meserette & Gwennael Gate
* Copyright (c) Aldebaran Robotics 2010
*/

#ifndef SOUNDPLAYBACK_H
#define SOUNDPLAYBACK_H

#include <boost/shared_ptr.hpp>
#include <alcommon/almodule.h>

namespace AL
{
  class ALBroker;
}

class ALSoundPlayback : public AL::ALModule
{
  struct wavheader
  {
    char chunkID[4];
    long chunkSize;
    char format[4];
    char subchunk1ID[4];
    long subchunk1Size;
    unsigned short audioFormat;
    unsigned short nbOfChannels;
    unsigned long sampleRate;
    unsigned long byteRate;
    unsigned short blockAlign;
    unsigned short bitsPerSample;
    char subchunk2ID[4];
    long subchunk2Size;
  };

public:
  ALSoundPlayback(boost::shared_ptr<AL::ALBroker> pBroker, const std::string& pName);

  virtual ~ALSoundPlayback();

  void playWavFile(const std::string &pFileName);

private:

  boost::shared_ptr <AL::ALProxy> audioDeviceProxy;
};
#endif

