# import cv2
import librosa
import matplotlib.pyplot as plt
import librosa.display as lid
import IPython.display as ipd
import numpy as np


def load_audio(filepath):
    audio, sr = librosa.load(filepath)
    return audio, sr


def show_image(filename):
    filepath = filepaths.external_dir / 'image_dataset' / f'{filename}.jpg'
    img = cv2.imread(filepath)
    img = img[...,::-1] # bgr => rgb
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('OFF')
    plt.show()
    return img


def get_spectrogram(audio, CFG):
    spec = librosa.feature.melspectrogram(y=audio, sr=CFG.sample_rate, 
                                   n_mels=CFG.img_size[0],
                                   n_fft=CFG.nfft,
                                   hop_length=CFG.hop_length,
                                   fmax=CFG.fmax,
                                   fmin=CFG.fmin,
                                   )
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec


def display_audio(row, CFG):
    caption = f'Id: {row.filename} | Name: {row.common_name} | Sci.Name: {row.scientific_name} | Rating: {row.rating}'
    
    audio, sr = load_audio(row.filepath)
    audio = audio[:CFG.audio_len]
    spec = get_spectrogram(audio, CFG)
    
    print("# Audio:")
    display(ipd.Audio(audio, rate=CFG.sample_rate))
    print("# Image:")
    # show_image(row.common_name)
    print('# Visualization:')
    fig, ax = plt.subplots(2, 1, figsize=(12, 2*3), sharex=True, tight_layout=True)
    fig.suptitle(caption)
    lid.waveshow(audio,
                 sr=CFG.sample_rate,
                 ax=ax[0])
    # Specplot
    lid.specshow(spec, 
                 sr = CFG.sample_rate, 
                 hop_length = CFG.hop_length,
                 n_fft=CFG.nfft,
                 fmin=CFG.fmin,
                 fmax=CFG.fmax,
                 x_axis = 'time', 
                 y_axis = 'mel',
                 cmap = 'coolwarm',
                 ax=ax[1])
    ax[0].set_xlabel('');
    fig.show()
