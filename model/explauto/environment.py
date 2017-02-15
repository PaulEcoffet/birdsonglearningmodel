import numpy as np

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment

from python_speech_features import mfcc

from synth import gen_sound

class BirdSongEnvironment(Environment):
    use_process = True

    def __init__(self, alphaf_shape=(1, 3), betaf_shape=(1, 1), nb_win=4,
                 nb_ceps=6, samplerate=44100):
        # m_size = duration + alphaf_dim + betaf_dim
        self.alphaf_shape = alphaf_shape
        self.betaf_shape = betaf_shape
        first_betaf_index = 1 + 2 * self.alphaf_shape[0] + alphaf_shape[1] * 3 + 1
        m_size = 1 + ((alphaf_shape[0] * 2 + alphaf_shape[1] * 3 + 1)
                  + (betaf_shape[0] * 2 + betaf_shape[1] * 3 + 1))

        self.nb_win = nb_win
        self.nb_ceps = nb_ceps
        s_size = 1 + self.nb_ceps * self.nb_win

        # Alpha motor bounds
        m_mins = [0] # duration min
        m_maxs = [1]
        for i in range(alphaf_shape[0]):
            m_mins.extend([0, 0]) # exp coef [a, b] for a*exp(-b * t)
            m_maxs.extend([10, 10])
        for i in range(alphaf_shape[1]):
            m_mins.extend([0, 0, 0]) # sin coef [a, p, f] for a*sin((p + t * 2pi)* (100*f))
            m_maxs.extend([2, 2*np.pi, 10])
        m_mins.append(-10) # constant
        m_maxs.append(10)
        # beta motor bounds
        for i in range(betaf_shape[0]):
            m_mins.extend([0, 0]) # exp coef [a, b] for a*exp(-b * t)
            m_maxs.extend([10, 10])
        for i in range(betaf_shape[1]):
            m_mins.extend([0, 0, 0]) # sin coef [a, p, f] for a*sin((p + t * 2pi)* (100*f))
            m_maxs.extend([2, 2*np.pi, 10])
        m_mins.append(-10) # constant
        m_maxs.append(10)

        s_mins = [0] + [-50] * (nb_win * nb_ceps)
        s_maxs = [1] + [50] * (nb_win * nb_ceps)

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.samplerate = samplerate

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):
        duration = m[0]
        commands = m[1:]
        try:
            res = self.listen(gen_sound(commands, duration * self.samplerate,
                                 alphaf_shape=self.alphaf_shape,
                                 betaf_shape=self.betaf_shape))
        except IndexError: # can happen with bad parameters in signal
            res = np.zeros((len(self.conf.s_maxs),))
        assert res.shape == np.array([len(self.conf.s_maxs)])
        return res

    def listen(self, sound):
        duration = len(sound) / self.samplerate
        winstep = duration / (self.nb_win + 1)
        winlen = winstep * 2
        return np.concatenate(
            (
                np.array([duration]),
                mfcc(sound,
                     self.samplerate,
                     winstep=winstep,
                     winlen=winlen,
                     numcep=self.nb_ceps+1
                     )[:self.nb_win, 1:].flatten()
            ))
