

1. python prepare_data.ipynb, which would
* Read XC data from xeno_canto_2023_audio_dir (specified in filepaths.yaml)
* Resample waveform to 32kHz and convert it to mono
* Save the processed audio data to .wav files - same xeno_canto_2023_audio_dir folder, but 'wav' subdir


2. python train_model.py, which would
* Read training data
* Train your model
* Save your model to 'model_dir' (specified in filepaths.yaml)


3. python pseudo114.py / pseudo168.py / pseudo226.py OR run make_pseudolabels.ipynb which would
* Load models states from 'model_dir' (specified in filepaths.yaml)
* Make pseudolabels for Training data/Additional XC data
* Save pseudolabels to 'model_dir', 'pseudolabels' subdir
* If an ensemble of models is used, pseudolabels will be saved to a new directory, named as a concatenation of model names, for example: 'exp114_exp122_exp145_exp148/pseudolabels'
