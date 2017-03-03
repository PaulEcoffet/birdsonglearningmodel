import subprocess
import numpy as np
from io import BytesIO

def exp_sin(x, p, nb_exp=2, nb_sin=2):
    ip = np.nditer(p)
    return np.sum([next(ip) * np.exp(-np.abs(next(ip)) * x)
                   for i in range(nb_exp)]
                  + [next(ip) * np.sin(next(ip) + (2 * np.pi * x) * next(ip))
                     for i in range(nb_sin)] + [next(ip)], axis=0)


def only_sin(x, p, nb_sin=3):
    ip = np.nditer(p)
    return np.sum([(next(ip) * x + next(ip))
                   * np.sin(next(ip) + (2*np.pi * x) * next(ip))
                   for i in range(nb_sin)] + [next(ip)], axis=0)


def f_str(p, nb_exp=2, nb_sin=2, x='x'):
    ip = np.nditer(p)
    out = ''
    for i in range(nb_exp):
        out += '{:.2} exp({:.2} {}) + '.format(float(next(ip)),
                                               -np.abs(float(next(ip))), x)
    for i in range(nb_sin):
        out += '{:.2f} sin(({:.2f} + 2π {})/{:.2f}) + '.format(float(next(ip)),
                                                               float(next(ip)),
                                                               x,
                                                               float(next(ip)))
    out += '{:.2f}'.format(float(next(ip)))
    return out


def gen_sound(params, length, falpha, fbeta, falpha_nb_args):
    alpha_beta = gen_alphabeta(params, length, falpha, fbeta, falpha_nb_args)
    out = synthesize(alpha_beta)
    return out


def gen_alphabeta(params, length, falpha, fbeta,
                  falpha_nb_args):
    # + 2 padding is necessary with ba synth.
    t = np.linspace(0, (length+2)/44100, length+2)
    alpha_beta = np.stack(
        (
            falpha(t, params[:falpha_nb_args]),
            fbeta(t, params[falpha_nb_args:])
        ), axis=-1)
    return alpha_beta


def synthesize(alpha_beta):
    """Return the song signal given the alpha beta parameters.

    alpha_beta - A 2d numpy.array of shape (length, 2)
                 with alpha on the alpha_beta[:, 0] elements
                 and beta on the alpha_beta[:, 1] elements

    Returns - 1D numpy.array with the normalized signal between -1 and 1
    """
    input_bytes = BytesIO()
    input_bytes.write(bytes(str(alpha_beta.shape[0]) + "\n", 'utf-8'))
    np.savetxt(input_bytes, alpha_beta)
    out_raw_call = subprocess.Popen(
        ["../../csynthesizer/alphabeta2dat"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    out_raw = out_raw_call.communicate(input=input_bytes.getvalue())[0]
    input_bytes.close()
    out = np.fromstring(out_raw, dtype=float, sep="\n")
    out = 2 * (out - np.nanmin(out)) / (np.nanmax(out) - np.nanmin(out)) - 1
    out[np.isnan(out)] = 0
    return out
