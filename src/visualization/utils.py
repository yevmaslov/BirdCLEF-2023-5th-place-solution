import librosa
import matplotlib.pyplot as plt
import librosa.display as lid
import IPython.display as ipd
import numpy as np


def display_audio(audio):
    display(ipd.Audio(audio, rate=32000))


def display_waveform(audio, ax):
    lid.waveshow(audio,
                 sr=32000,
                 ax=ax)


def display_spec(spec, ax):
    spec = spec[0]
    spec = spec.numpy()
    im = ax.imshow(librosa.power_to_db(spec), origin="lower", aspect="auto")
