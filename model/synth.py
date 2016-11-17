import subprocess
import numpy as np
from io import BytesIO

def f(x, p, nb_exp=2, nb_sin=2):
    ip = np.nditer(p)
    x = x/22500
    return np.sum([next(ip) * np.exp(-next(ip) * x) for i in range(nb_exp)]
                + [next(ip) * np.sin((next(ip) + 2*np.pi * x) / next(ip)) for i in range(nb_sin)] + [next(ip)], axis=0)

def gen_sound(params, length, alphaf_shape=(1, 3), betaf_shape=(0, 1)):
    t = np.arange(0, length+2)  # + 2 padding is necessary with ba synth.
    nb_args_alpha = alphaf_shape[0] * 2 + alphaf_shape[1] * 3 + 1
    alpha_beta = np.stack((
        f(t, params[:nb_args_alpha], alphaf_shape[0], alphaf_shape[1]),
        f(t, params[nb_args_alpha:], betaf_shape[0], betaf_shape[1])), axis=-1
    )
    out = synthesize(alpha_beta)
    return out

def synthesize(alpha_beta, amplitudeforwav=False):
    input_bytes = BytesIO()
    input_bytes.write(bytes(str(len(alpha_beta)) + "\n", 'utf-8'))
    np.savetxt(input_bytes, alpha_beta)
    out_raw_call = subprocess.Popen(["../csynthesizer/alphabeta2dat"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE)
    out_raw = out_raw_call.communicate(input=input_bytes.getvalue())[0]
    input_bytes.close()
    out = np.fromstring(out_raw, dtype=float, sep="\n")
    if amplitudeforwav:
        out = dat2wav(out)
    return out

def dat2wav(dat):
    ampMax = np.max(dat)
    ampMin = np.min(dat)
    if ampMax - ampMin == 0:
        scalingFactor = 1
    else:
        scalingFactor = 65535 / (ampMax - ampMin)
    wav = np.clip(np.round((dat - ampMin) * scalingFactor) - 32768, -32768, 32767).astype(np.dtype('int16'))
    return wav
