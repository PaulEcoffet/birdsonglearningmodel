import unittest
import numpy as np
from synth import synthesize, dat2wav

from synth.alphabeta2dat import AlphaBeta2Dat
from synth.dat2wav import Dat2Wav

class TestSynth(unittest.TestCase):

    def test_synth(self):
        alpha_beta = np.loadtxt('test/comparison_files/test1_ab.dat')
        out = synthesize(alpha_beta, False)
        np.testing.assert_array_equal(out, np.loadtxt('test/comparison_files/test1_out.dat'))
