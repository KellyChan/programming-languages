"""
Author: Kelly Chan
Date: July 20 2014
"""


import pygame
import numpy as np
from scipy.io import wavfile


def speedx(sound_array, factor):
    """ speeds up / slows down a sound, by some factor """

    #  the changes of the sound = sound_array * factor
    indices = np.round(np.arange(0, len(sound_array), factor))
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[indices]


def stretch(sound_array, factor, window_size, h):
    """ stretches / shortens a sound, by some factor """

    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(len(sound_array) / factor + window_size)

    for i in np.arange(0, len(sound_array) - (window_size+h), h*factor):

        #  two potentially overlapping subarrays
        a1 = sound_array[i : i+window_size]
        a2 = sound_array[i+h : i+window_size+h]

        #  the spectra of these arrays
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)

        #  rephase all frequencies
        phase = (phase + np.angle(s2/s1)) % 2 * np.pi

        a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
        i2 = int(i / factor)
        result[i2 : i2+window_size] += hanning_window * a2_rephased

    result = ((2**(16-4)) * result / result.max())  # normalize (16 bit)

    return result.astype('int16')


def pitchshift(sound_array, n, window_size=2**13, h=2**11):
    """ changing the pitch of a sound by ``n`` semitones """

    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(sound_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)


def main():

    sources = "G:/vimFiles/python/projects/piano-keybroad/sources/"

    #  init
    fps, bowl_sound = wavfile.read(sources + "bowl.wav")
    pygame.mixer.init(fps, -16, 1, 100)  # flexible
    screen = pygame.display.set_mode((640, 480))  # for the focus

    #  set keys
    #keys = open(sources + "typewriter.kb").read().split('\n')
    keys = open(sources + "my_keyboard.kb").read().split('\n')

    #  set sounds
    tones = range(-25, 25)
    transposed_sounds = [pitchshift(bowl_sound, n) for n in tones]
    sounds = map(pygame.sndarray.make_sound, transposed_sounds)

    #  play
    key_sound = dict(zip(keys, sounds))
    is_playing = {k: False for k in keys}
    while True:

        event = pygame.event.wait()

        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key = pygame.key.name(event.key)

        if event.type == pygame.KEYDOWN:
            
            if (key in key_sound.keys()) and (not is_playing[key]):
                key_sound[key].play(fade_ms=50)
                is_playing[key] = True

            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise KeyboardInterrupt

        elif event.type == pygame.KEYUP and key in key_sound.keys():
            key_sound[key].fadeout(50)  # stops with 50ms fadeout
            is_playing[key] = False




if __name__ == '__main__':
    main()

