import subprocess
import numpy as np
from io import BytesIO


def gen_sound(params, length, samplerate=44100):
    t = np.linspace(0, length/samplerate, length+2)  # + 2 padding is necessary with ba synth.
    a, b, c, d, e, f, g, h, i, j, k = params
    alpha_beta = np.stack((
        a*np.exp(-b*t) + c * np.cos(d * 2 * np.pi * t + e) + f,
        g*t + h + i*np.cos(j * 2 * np.pi * t + k)), axis=-1
    )
    out = synthesize(alpha_beta)
    return out

def synthesize(alpha_beta, amplitudeforwav=False):
    input_bytes = BytesIO()
    input_bytes.write(bytes(str(len(alpha_beta)) + "\n", 'utf-8'))
    np.savetxt(input_bytes, alpha_beta)
    out_raw = subprocess.run(["../csynthesizer/alphabeta2dat"],
                    input=input_bytes.getvalue(),
                    stdout=subprocess.PIPE).stdout
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
