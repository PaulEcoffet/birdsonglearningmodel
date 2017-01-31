import numpy as np

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment

from python_speech_features import mfcc

from synth import gen_sound

class BirdSongEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, samplerate):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.samplerate = samplerate

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):
        duration = m[0]
        commands = m[1:]
        winstep = duration / 10
        winlen = winstep * 2
        return [duration] + \
                mfcc(
                    gen_sound(commands, duration * self.samplerate,
                              alphaf_shape=(1, 3),
                              betaf_shape=(1, 1)),
                    self.samplerate,
                    winstep=winstep,
                    winlen=winlen,
                    numcep=8
                )[..., 1:].flatten()
