/**
* @author Vincent Meserette & Gwennael Gate
* Copyright (c) Aldebaran Robotics 2010
*/

#include "alsoundplayback.h"

#include <iostream>
#include <alcommon/alproxy.h>
#include <boost/shared_ptr.hpp>
#include <alcommon/albroker.h>
#include <alcommon/almodule.h>

using namespace AL;

ALSoundPlayback::ALSoundPlayback(boost::shared_ptr<ALBroker> pBroker, const std::string& pName ): ALModule(pBroker, pName )
{
  setModuleDescription("An example of how to send sound onto the nao's loudspeakers.");

  /// Define callable methods with there description
  functionName("playWavFile",
               getName(),
               "sends the content of a wav file of a remote desktop onto the nao's loudspeakers ");

  addParam("pFileName", "Name of twav file to play");
  BIND_METHOD(ALSoundPlayback::playWavFile);

  audioDeviceProxy = getParentBroker()->getProxy("ALAudioDevice");
}

ALSoundPlayback::~ALSoundPlayback()
{

}

void ALSoundPlayback::playWavFile(const std::string &pFileName)
{
  /// Maximum buffer size that ALAudioDevice can send
  const int outputBufferSize = 16384;
  /// Number of output channels on Nao
  const int numberOfOutputChannels = 2;
  FILE *fInputWavFile;

  /// Opening of the input file
  if((fInputWavFile = fopen(pFileName.c_str(), "rb" )) == NULL)
  {
    throw AL::ALError("ALSoundProcessing",
                      "playWavFile()",
                      "Failed to open input sound file.");
  }

  /// Reading of the wav header
  wavheader * wavHeader = new wavheader;
  fread(wavHeader,1,44,fInputWavFile);

  unsigned short nbOfChannels = wavHeader->nbOfChannels;
  unsigned long sampleRate = wavHeader->sampleRate;
  unsigned short bitsPerSample = wavHeader->bitsPerSample;

  try
  {
    /// Set output sample rate of audiodevice
    audioDeviceProxy->callVoid("setParameter",
                               std::string("outputSampleRate"),
                               (int)sampleRate);
  }
  catch(ALError &e)
  {
    throw AL::ALError("ALSoundProcessing",
                      "playWavFile()",
                      e.getDescription());
  }

  /// Go to the beginning of the audio data
  fseek(fInputWavFile, 44, SEEK_SET);

  /// Reading of the audio data
  /// Buffer to store the data contained in the wav file
  short * fInputAudioData = new short[outputBufferSize*nbOfChannels];
  /// Buffer to construct stereo audio data
  short * fStereoAudioData = new short[outputBufferSize*numberOfOutputChannels];

  /// ALValue where to store the stereo audio data to send
  ALValue pDataBin;
  while(!feof(fInputWavFile))
  {
    /// Read samples from file
    int fNbOfInputSamples = fread(fInputAudioData,
                                  bitsPerSample/8*nbOfChannels,
                                  outputBufferSize,
                                  fInputWavFile);

    if(nbOfChannels ==1)
    {
      /// Construction of stereo audio data in case the input file is mono
      int i=0;
      for(int j=0 ; j<fNbOfInputSamples ; j++)
      {
        fStereoAudioData[i] = fInputAudioData[j];
        fStereoAudioData[i+1] = fInputAudioData[j];
        i += numberOfOutputChannels;
      }

      /// Transform audio data to binary samples
      pDataBin.SetBinary(fStereoAudioData,
                         fNbOfInputSamples*sizeof(short)*numberOfOutputChannels);
    }
    else if(nbOfChannels == 2)
    {
      /// Transform audio data to binary samples
      pDataBin.SetBinary(fInputAudioData,
                         fNbOfInputSamples*sizeof(short)*numberOfOutputChannels);
    }
    else
    {
      throw AL::ALError("ALSoundProcessing",
                        "playWavFile()",
                        "The wav file should contain a mono or stereo signal.");
    }

    /// Send the stereo audio data to audiodevice module
    try
    {
      audioDeviceProxy->call<bool>("sendRemoteBufferToOutput",
                                   fNbOfInputSamples,
                                   pDataBin);
    }
    catch(AL::ALError &e)
    {
      throw AL::ALError("ALSoundProcessing","playWavFile()", e.getDescription());
    }
  }
  fclose(fInputWavFile);
}


