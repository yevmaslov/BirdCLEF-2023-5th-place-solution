import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn

from models.audio_transforms import *
from models.spec_transforms import *
from nnAudio import Spectrogram


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, wav, label, weight, sr=32000):
        for trns in self.transforms:
            wav, label, weight, sr = trns(wav, label, weight, sr)
        return wav, label, weight, sr
    
    
class OneOf(Compose):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, wav, label, weight, sr=32000):
        data = wav
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data, label, weight, sr = t(wav, label, weight, sr)
        return data, label, weight, sr


def get_noise_augmentations(config):
    noise_augmentations = []

    if config.dataset.use_gaussian_noise:
        noise_augmentations.append(GaussianNoise(p=1, min_snr=5, max_snr=20))
    if config.dataset.use_pink_noise:
        noise_augmentations.append(PinkNoise(p=1, min_snr=5, max_snr=20))
    if config.dataset.use_brown_noise:
        noise_augmentations.append(BrownNoise(p=1, min_snr=5, max_snr=20))
    if config.dataset.use_noise_injection:
        noise_augmentations.append(NoiseInjection(p=1, max_noise_level=0.04))
    if config.dataset.use_esc50_noise:
        noise_augmentations.append(BackgroundNoise(config.dataset.esc50_df, config.dataset.train_audio_len, p=1))
    if config.dataset.use_no_call_noise:
        noise_augmentations.append(BackgroundNoise(config.dataset.no_call_df, config.dataset.train_audio_len, p=1))
    
    noise_augmentations = OneOf(noise_augmentations, p=config.dataset.noise_aug_prob)
    return [noise_augmentations]


def get_audio_transforms(config, train=True):
    audio_transforms = []

    if train:
        pad_truncate = PadTruncate(p=1, max_len=config.dataset.train_audio_len)
        noise_transforms = get_noise_augmentations(config)
        mix_up = MixUp(config.dataset.mixup_df, config, p=config.dataset.audio_mix_up_prob)
        normalize = Normalize(p=1)
        random_volume = RandomVolume(p=0.5)

        audio_transforms.append(pad_truncate)
        # audio_transforms.append(random_volume)
        audio_transforms.append(mix_up)
        audio_transforms.extend(noise_transforms)
        audio_transforms.append(normalize)

    else:
        pad_truncate = PadTruncate(p=1, max_len=config.dataset.train_audio_len)
        normalize = Normalize(p=1)

        audio_transforms.append(pad_truncate)
        audio_transforms.append(normalize)
    return Compose(audio_transforms)


def get_spectrogram_augmentations(config):
    spectrogram_augmentations = []

    if config.dataset.time_mask_prob > 0:
        spectrogram_augmentations.append(TimeMasking(p=config.dataset.time_mask_prob))
    if config.dataset.freq_mask_prob > 0:
        spectrogram_augmentations.append(FrequencyMasking(p=config.dataset.freq_mask_prob))
    if config.dataset.second_time_mask_prob > 0:
        spectrogram_augmentations.append(TimeMasking(p=config.dataset.second_time_mask_prob))

    return spectrogram_augmentations


def get_spectrogram_transforms(config, train=True):
    spectrogram_transforms = []

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        32000,
        n_mels=config.dataset.n_mels,
        n_fft=config.dataset.nfft,
        hop_length=config.dataset.hop_length,
        f_max=config.dataset.fmax,
        f_min=config.dataset.fmin,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=config.dataset.top_db)
    spec_norm = NormalizeMelSpec()

    spectrogram_transforms.append(mel_spectrogram)
    spectrogram_transforms.append(amplitude_to_db)
    spectrogram_transforms.append(spec_norm)

    if train:
        spectrogram_augmentations = get_spectrogram_augmentations(config)
        spectrogram_transforms.extend(spectrogram_augmentations)

    return nn.Sequential(*spectrogram_transforms)
