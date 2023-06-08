Hello!

Below you can find an outline of how to reproduce my solution for the BirdCLEF-2023 competition.
If you run into any trouble with the setup/code or have any questions please contact me at sqrt.evmaslov@gmail.com



# HARDWARE: (The following specs were used to create the original solution)

Ubuntu 20.04 LTS

1xNVIDIA A6000 or 1xNVIDIA A100 (depending on availability)

12vCPU, 45 or 90 GB memory


# SOFTWARE (python packages are detailed separately in `requirements.txt`):

Python 3.5.1

CUDA 11.6

nvidia drivers v.510.73.05



# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)


mkdir data

mkdir data/raw

mkdir data/external

mkdir data/external/birdclef-2023-xc

mkdir data/processed

mkdir models

kaggle competitions download -c birdclef-2023

unzip birdclef-2023.zip -d data/raw

kaggle competitions download -c birdclef-2022

unzip birdclef-2022.zip -d data/external/birdclef-2022

kaggle competitions download -c birdclef-2021

unzip birdclef-2021.zip -d data/external/birdclef-2021

kaggle datasets download -d evgeniimaslov2/birdclef-2023-xc-primary

unzip birdclef-2023-xc-primary.zip -d data/external/birdclef-2023-xc

kaggle datasets download -d evgeniimaslov2/birclef-2023-xc-secondary

unzip birclef-2023-xc-secondary.zip -d data/external/birdclef-2023-xc

kaggle datasets download -d evgeniimaslov2/birdclef-2023-xc-africa-1

unzip birdclef-2023-xc-africa-1.zip -d data/external/birdclef-2023-xc

kaggle datasets download -d evgeniimaslov2/birdclef-2023-xc-africa-2

unzip birdclef-2023-xc-africa-2.zip -d data/external/birdclef-2023-xc

kaggle datasets download -d evgeniimaslov2/birdclef-2023-xc-africa-3

unzip birdclef-2023-xc-africa-3.zip -d data/external/birdclef-2023-xc

kaggle datasets download -d evgeniimaslov2/birdclef-2023-xc-africa-4

unzip birdclef-2023-xc-africa-4.zip -d data/external/birdclef-2023-xc

kaggle datasets download -d evgeniimaslov2/birdclef-2023-xc-africa-5

unzip birdclef-2023-xc-africa-5.zip -d data/external/birdclef-2023-xc


# DATA PROCESSING

Run notebooks/prepare_xc_data.ipynb - this will resample XC waveform to mono, 32kHz, and save processed sample as a .wav file.


# MODEL BUILD

Run notebooks/train_from_scratch.sh to retrain models from scratch.

It should take ~3 weeks to train all models and get the final solution.



