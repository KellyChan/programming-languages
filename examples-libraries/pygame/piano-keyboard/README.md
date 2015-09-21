Piano Keyboard
==============

### Algorithms

speed/pitch = frequencies * factor
- factor = 2^(n/12), 12 is the 12 semitones in a octave
- stretch/frequencies = basic tones -> two overlapped subarrays -> reshaped -> normalized


| Function | Variable     | Algorithm                                                         |
|:---------|:-------------|:------------------------------------------------------------------|
| speed    | pitch        | frequencies * factor                                              |
| stretch  | frequencies  | basic tones -> two overlapped subarrays -> reshaped -> normalized |
|          | factor       | 2^(n/12), 12 is the 12 semitones in a octave                      |



### Steps

- step1. get a basic tone
- step2. pitch shifting (from 1 to 50) / Algorithms
- step3. map the tones to the keys (keyboard)
- step4. set up to play
  

### Source Codes

- set up the keyboard (keys)
- set up the tones and combine them to the keys


### Reference
1. Python, Pitch Shifting, and the Pianoputer: [English][en], [Chinese][cn]

  [en]: http://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/
  [cn]: http://blog.jobbole.com/72745/
