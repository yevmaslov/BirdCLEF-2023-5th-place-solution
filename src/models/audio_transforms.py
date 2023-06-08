import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import colorednoise as cn
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")


def get_sample(row, config, train):
    subset = row['subset']
    fp = row['filepath']
    fn = row['filename']

    if subset == 'zenodo':
        fn = '_'.join(str(fp.name).split('_')[:-1])
        sample = config.full_zenodo_data[fn].sample(1)
        duration = sample['duration']
        weight = sample['weight']

    elif subset == 'pseudo':
        sample = config.full_train23_pseudo[fn].sample(1)
        # sample = config.full_train23_pseudo.loc[[fn]].sample(1)
        sample = sample.iloc[0]

        duration = sample['duration']
        weight = sample['weight']
        
        inference_type = config.dataframe.pseudo_2023_config.inference_type
        assert inference_type in ['window', 'sample', 'frame']
        
        if inference_type == 'window':
            start = int(sample['start'] / 32000)
            stop = int(sample['end'] / 32000)
            label = sample[config.dataset.labels]
        elif inference_type == 'sample':
            raise NotImplementedError
        else:
            raise NotImplementedError

    elif subset == 'pseudo_xc':
        sample = config.full_train23_xc_pseudo[fn].sample(1)
        # sample = config.full_train23_xc_pseudo.loc[[fn]].sample(1)
        sample = sample.iloc[0]

        duration = sample['duration']
        weight = sample['weight']
        
        inference_type = config.dataframe.pseudo_2023_xc_config.inference_type
        assert inference_type in ['window', 'sample', 'frame']

        if inference_type == 'window':
            start = int(sample['start'] / 32000)
            stop = int(sample['end'] / 32000)
            label = sample[config.dataset.labels]
        elif inference_type == 'sample':
            fn = fn.replace('.wav', '.npy')
            label_fp = f'../models/exp143/pseudolabels_xc_frame/fold{config.fold}/{fn}'
            label = load_labels(label_fp, 0, duration)
            start, stop = get_start_and_stop_seconds(audio_duration=duration, window_size=config.dataset.train_duration, train=train)
        else:
            fn = fn.replace('.wav', '.npy')
            label_fp = f'../models/exp143/pseudolabels_xc_frame/fold{config.fold}/{fn}'
            start, stop = get_start_and_stop_seconds(audio_duration=duration, window_size=config.dataset.train_duration, train=train)
            label = load_labels(label_fp, start, stop)
            
    elif subset == 'africa':
        sample = config.full_africa[fn].sample(1)
        # sample = config.full_africa.loc[[fn]].sample(1)
        sample = sample.iloc[0]

        duration = sample['duration']
        weight = sample['weight']

        start = int(sample['start'] / 32000)
        stop = int(sample['end'] / 32000)
        label = sample[config.dataset.labels]

    else:
        sample = row.copy()
        duration = sample['duration']
        weight = sample['weight']
        label = sample[config.dataset.labels]
        start, stop = get_start_and_stop_seconds(audio_duration=duration, window_size=config.dataset.train_duration, train=train)
        
    return fp, start, stop, label, weight



def get_start_and_stop_seconds(audio_duration, window_size, train=False):
    if train and audio_duration > window_size:
        high = max(1, audio_duration - window_size)
        start = np.random.uniform(low=0, high=high)
        stop = start + window_size
    else:
        start = 0
        stop = start + window_size
    return int(start), int(stop)


def load_sample_wav_sf(fp, start, stop):
    wav, sr = sf.read(fp, start=start*32000, stop=stop*32000, dtype='float32')
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav)
    wav = wav.astype('float32')
    return wav, sr


def load_labels(fp, start, stop):
    labels = np.load(fp)
    labels = labels[start*2:stop*2, :]
    labels = labels.max(axis=0)
    return labels


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, wav, label, weight, sr):
        if self.always_apply:
            return self.apply(wav, label, weight, sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(wav, label, weight, sr)
            else:
                return wav, label, weight, sr

    def apply(self, wav, label, weight, sr=32000):
        raise NotImplementedError


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, wav, label, weight, sr=32000):
        max_vol = np.abs(wav).max()
        y_vol = wav * 1 / max_vol
        return np.asfortranarray(y_vol), label, weight, sr


class PadTruncate(AudioTransform):
    def __init__(self, max_len, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.max_len = max_len

    def apply(self, wav, label, weight, sr=32000):
        wav_len = len(wav)
        if wav_len < self.max_len:
            diff = (self.max_len // wav_len) + 1
            wav = np.concatenate([wav] * diff, axis=0)
        wav = wav[:self.max_len]
        return wav, label, weight, sr


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, wav, label, weight, sr=32000):
        y_mm = wav - wav.mean()
        out = y_mm / y_mm.abs().max()
        return out, label, weight, sr


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, wav, label, weight, sr=32000):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(wav))
        augmented = (wav + noise * noise_level).astype(wav.dtype)
        return augmented, label, weight, sr


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, wav, label, weight, sr=32000):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(wav ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(wav))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (wav + white_noise * 1 / a_white * a_noise).astype(wav.dtype)
        return augmented, label, weight, sr


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, wav, label, weight, sr=32000):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(wav ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(wav))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (wav + pink_noise * 1 / a_pink * a_noise).astype(wav.dtype)
        return augmented, label, weight, sr


class BrownNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, wav, label, weight, sr=32000):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(wav ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        brown_noise = cn.powerlaw_psd_gaussian(2, len(wav))
        a_brown = np.sqrt(brown_noise ** 2).max()
        augmented = (wav + brown_noise * 1 / a_brown * a_noise).astype(wav.dtype)
        return augmented, label, weight, sr


class BackgroundNoise(AudioTransform):
    def __init__(self, noise_df, max_len, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__()

        self.noise_df = noise_df
        self.max_len = max_len
        self.p = p
        self.always_apply = always_apply
        self.pad_truncate = PadTruncate(self.max_len, p=1)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def get_random_sample(self):
        idx = np.random.randint(0, len(self.noise_df))
        fp = self.noise_df.iloc[idx].filepath
        duration = self.noise_df.iloc[idx].duration
        
        start, stop = get_start_and_stop_seconds(audio_duration=duration, window_size=self.max_len, train=True)
        wav, _ = load_sample_wav_sf(fp, start=start, stop=stop)
        wav, _, _, _ = self.pad_truncate(wav, None, None, sr=32000)
        return wav

    def apply(self, wav, label, weight, sr=32000):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(wav ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        random_noise = self.get_random_sample()
        a_random = np.sqrt(random_noise ** 2).max()
        augmented = (wav + random_noise * 1 / a_random * a_noise).astype(wav.dtype)
        return augmented, label, weight, sr


class ShiftTime(AudioTransform):
    def __init__(self, max_len, always_apply=False, p=0.5):
        self.max_len = max_len
        self.p = p
        self.always_apply = always_apply

    def apply(self, wav, label, weight, sr=32000):
        n_max = max(1, len(wav) - self.max_len)
        shift = np.random.randint(0, n_max)
        wav = wav[shift:shift + self.max_len]
        return wav, label, weight, sr


class ChangeSpeed:
    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply

    def apply(self, wav, label, weight, sr=32000):
        speed_factor = np.random.choice([0.9, 1, 1.1])
        if speed_factor == 1:
            return wav
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(sr)],
        ]
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav, sr, sox_effects
        )
        return audio, label, weight, sr


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, wav, label, weight, sr=32000):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(wav, sr=32000, n_steps=n_steps)
        return augmented, label, weight, sr


class MixUp(AudioTransform):
    def __init__(self, dataframe, config, always_apply=False, p=0.5):
        self.dataframe = dataframe
        self.config = config
        self.p = p
        self.audio_len = config.dataset.train_audio_len
        self.pad_truncate = PadTruncate(self.audio_len, p=1)
        self.always_apply = always_apply

    def get_random_sample(self):
        idx = np.random.randint(0, len(self.dataframe))
        row = self.dataframe.iloc[idx]
        
        fp, start, stop, label, weight = get_sample(row, self.config, True)

        wav, _ = load_sample_wav_sf(fp, start=start, stop=stop)
        wav, _, _, _ = self.pad_truncate(wav, None, None, sr=32000)

        return wav, label, weight

    def apply(self, wav, label, weight, sr=32000):
        random_wav, random_label, random_weight = self.get_random_sample()

        alpha = self.config.dataset.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        wav = lam * wav + (1 - lam) * random_wav
        label = lam * label + (1 - lam) * random_label
        weight = lam * weight + (1 - lam) * random_weight
        return wav, label, weight, sr


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, wav, label, weight, sr=32000):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(wav, rate)
        return augmented, label, weight, sr


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, wav, label, weight, sr=32000):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(wav, db), label, weight, sr
        else:
            return volume_down(wav, db), label, weight, sr


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, wav, label, weight, sr=32000):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs, label, weight, sr