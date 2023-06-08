import torch
from torch.utils.data import Dataset
import librosa
import torchaudio
from torchaudio import transforms
import numpy as np
import torch.nn as nn
from models.audio_transforms import load_sample_wav_sf, get_start_and_stop_seconds, load_labels, get_sample
from models.transforms_utils import get_audio_transforms, get_spectrogram_transforms



class CustomDataset(Dataset):
    def __init__(self, dataframe, config, train=True):
        self.dataframe = dataframe
        self.config = config
        
        self.train = train
        self.audio_len = config.dataset.train_audio_len
        self.hop_length = config.dataset.hop_length

        self.audio_transforms = get_audio_transforms(config, train)
        self.spectrogram_transforms = get_spectrogram_transforms(config, train)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        fp, start, stop, label, weight = get_sample(row, self.config, self.train)
        
        raw_waveform, sr = load_sample_wav_sf(fp, start=start, stop=stop)
        
        raw_waveform = np.nan_to_num(raw_waveform)
        wav, label, weight, sr = self.audio_transforms(raw_waveform, label, weight, sr)
        wav = np.nan_to_num(wav)

        spec = self.spectrogram_transforms(torch.tensor(wav).view(1, len(wav)))
        spec = np.nan_to_num(spec)

        if self.config.model.architecture == 'psi_cnn':
            spec = torch.tensor(spec)
            spec = spec[:, :, :1872]
            spec = spec.reshape((1, 128, 6, 1872 // 6))
            spec = spec.permute(2, 0, 1, 3)

        output = {
            'waveform': torch.tensor(wav),
            'spec': spec,
            'labels': torch.tensor(label, dtype=torch.float),
            'weight': torch.tensor(weight)
        }
        return output
