import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from transformers import AutoTokenizer
import json
import os
import sys
sys.path.append('../src')

from environment.utils import load_filepaths, load_config, seed_everything, create_run_folder, save_config
from logger.wandb import init_wandb
from features.text import get_tokenized_text
from models.dataset import CustomDataset
from models.model import CustomModel
from models.trainer import Trainer

from transformers import DataCollatorWithPadding
from tqdm.notebook import tqdm

from models.utils import get_valid_steps
from models.optimizer import get_optimizer
from models.scheduler import get_scheduler
from logger.logger import Logger


import torchaudio
from IPython.display import Audio

from tqdm.notebook import tqdm
tqdm.pandas()
import warnings

import torch.nn.functional as F

from models.dataset import CustomDataset
from models.model import TattakaModel

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from data.load_data import *
from pathlib import Path

from models.spec_transforms import *
from models.audio_transforms import *
from models.transforms_utils import *
import random

import timm

device = "cuda" if torch.cuda.is_available() else "cpu"


from models.utils import batch_to_device

def validate(model, dataloader):
    model.eval()

    predictions = []
    frame_predictions = []

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(device)

        with torch.no_grad():
            y_pred, output_frame = model(batch)
        
        logits = y_pred.detach().to('cpu').numpy()
        output_frame = torch.sigmoid(output_frame).detach().to('cpu').numpy()
        
        predictions.append(logits)
        frame_predictions.append(output_frame)

    predictions = np.concatenate(predictions)
    frame_predictions = np.concatenate(frame_predictions)
    return predictions, frame_predictions

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, config):
        self.dataframe = dataframe
        self.config = config
        self.sr = self.config.dataset.sample_rate

        self.audio_transforms = get_audio_transforms(config, False)
        self.spectrogram_transforms = get_spectrogram_transforms(config, False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        fp = self.dataframe.iloc[item].path
        duration = self.dataframe.iloc[item].duration
        start = self.dataframe.iloc[item].start
        stop = self.dataframe.iloc[item].end

        audio, sr = self.load_sample_wav(fp, start, stop)
        audio = audio.astype('float32')
        audio = np.nan_to_num(audio)
        
        audio, _, _, _ = self.audio_transforms(audio, None, None, None)
        audio = np.nan_to_num(audio)

        spec = self.spectrogram_transforms(torch.tensor(audio).view(1, len(audio)))
        # spec = np.nan_to_num(spec)
        return spec

    def load_sample_wav(self, fp, start, stop):
        wav, sr = sf.read(fp, start=start, stop=stop)
        if len(wav) > 1:
            wav = librosa.to_mono(wav)
        return wav, sr

def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class AttHead(nn.Module):
    def __init__(
        self, in_chans, p=0.5, num_class=264, train_period=15.0, infer_period=5.0
    ):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)

        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)
        # print(feat.shape)
        
        time_att = torch.tanh(self.attention(feat))
        
        assert self.train_period >= self.infer_period
        
        if self.training or self.train_period == self.infer_period: # or True
            # print('train')
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)
            logits = torch.sum(
                self.fix_scale(feat) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
        else:
            # print('eval')
            clipwise_pred_long = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)
            
            feat_time = feat.size(-1)
            start = feat_time / 2 - feat_time * (self.infer_period / self.train_period) / 2
            end = start + feat_time * (self.infer_period / self.train_period)
            
            start = int(start)
            end = int(end)
            
            feat = feat[:, :, start:end]
            att = torch.softmax(time_att[:, :, start:end], dim=-1)
            
            # print(feat.shape)
            
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * att,
                dim=-1,
            )
            logits = torch.sum(
                self.fix_scale(feat) * att,
                dim=-1,
            )
            time_att = time_att[:, :, start:end]
        return (
            logits,
            clipwise_pred,
            self.fix_scale(feat).permute(0, 2, 1),
            time_att.permute(0, 2, 1),
        )

    
def get_timm_backbone(config, pretrained):
    backbone = timm.create_model(
        config.model.backbone_type,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        in_chans=1,
    )
    return backbone


class TattakaModel(nn.Module):
    def __init__(
        self,
        config,
        pretrained,
    ):
        super().__init__()

        # self.model = get_timm_backbone(config, pretrained)
        self.model = timm.create_model(
            config.model.backbone_type, features_only=True, pretrained=False, in_chans=1
        )
        encoder_channels = self.model.feature_info.channels()
        dense_input = encoder_channels[-1]
        self.head = AttHead(
            dense_input,
            p=config.model.dropout,
            num_class=len(config.dataset.labels),
            train_period=config.dataset.train_duration,
            infer_period=config.dataset.valid_duration,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, images):
        spec = images
        
        feats = self.model(spec)
        logits, output_clip, output_frame, output_attention = self.head(feats[-1])
        
        return output_clip, output_frame


import sklearn.metrics

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score


def map_score(solution, submission):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score


filepaths = load_filepaths('../filepaths.yaml')
config = load_config('../config.yaml')
config.dataset.train_duration = 5
config.dataset.valid_duration = 5

train = pd.read_csv(filepaths.raw_dir / 'train_metadata.csv')
folds = pd.read_csv(filepaths.processed_dir / 'birdclef-2023' / 'folds.csv')
duration = pd.read_csv(filepaths.processed_dir / 'birdclef-2023' / 'duration.csv')

labels = sorted(train.primary_label.unique())
config.dataset.labels = labels

train = pd.merge(train, folds, on='url', how='left')
train = pd.merge(train, duration, on='url', how='left')

train_audio_dir = filepaths.raw_dir / 'train_audio'
train['filepath'] = train_audio_dir / train['filename']

train_temp = train.copy()

for label in labels:
    primary = train['primary_label'] == label
    secondary = train['secondary_labels'].apply(lambda x: label in x)
    train[label] = (primary | secondary).astype(int)
    
train["weight"] = np.clip(train["rating"] / train["rating"].max(), 0.1, 1.0)
train['filepath'] = filepaths.raw_dir / 'train_audio' / train['filename']

xc_2023_data = load_xc_data(config.dataframe.xc_2023_data_config, filepaths, year=2023)


year = 2023

xc_data = pd.read_csv(filepaths.processed_dir / f'birdclef-{year}_XenoCanto' / 'metadata.csv')
xc_data['filename'] = xc_data['filename'].str.replace('.mp3', '.wav')

fp = Path('../data/external/birdclef-2023-africa/wav')
xc_data['filepath'] = fp / xc_data['filename']

xc_data['rating'].fillna(1, inplace=True)

# if 'duration' in xc_data.columns:
#     xc_data.drop(columns=['duration'], inplace=True)
    
xc_data = xc_data[xc_data['primary_label'] == 'africa'].reset_index(drop=True)

# duration = pd.read_csv('../data/external/birdclef-2023-africa/duration.csv')

# xc_data['url'] = xc_data['url'].apply(preprocess_url)
# duration['url'] = duration['url'].apply(preprocess_url)

# xc_data = pd.merge(xc_data, duration, on=['url', 'filename'], how='left')
# xc_data = xc_data[xc_data['duration'] >= 1].reset_index(drop=True)

xc_data['fold'] = 99

xc_data = xc_data.reset_index(drop=True)
xc_data['duration'] = xc_data['duration'].astype(int)
xc_data['duration_float'] = xc_data['duration']

print(f'XenoCanto {year} shape: ', xc_data.shape)

def update_max_duration(df, max_duration=180):
    df['md'] = max_duration
    df['duration'] = df[['duration', 'md']].min(axis=1)
    df = df.drop('md', axis=1)
    return df

xc_2023_data = update_max_duration(xc_2023_data, max_duration=120)
accepted_licenses = [
    # '//creativecommons.org/licenses/by-nc-nd/4.0/',
    '//creativecommons.org/licenses/by-nc-sa/4.0/',
    # '//creativecommons.org/licenses/by-nc-nd/2.5/',
    # '//creativecommons.org/licenses/by-nc-nd/3.0/',
    '//creativecommons.org/licenses/by-nc-sa/3.0/',
    # '//creativecommons.org/publicdomain/zero/1.0/',
    '//creativecommons.org/licenses/by-sa/4.0/',
    '//creativecommons.org/licenses/by/4.0/',
    '//creativecommons.org/licenses/by-nc/4.0/',
    '//creativecommons.org/licenses/by-sa/3.0/',
]

xc_data['fn'] = xc_data['filename'].apply(lambda x: x.split('_')[0])
xc_data = xc_data[xc_data['license'].isin(accepted_licenses)].reset_index(drop=True)

xc_data.loc[xc_data['duration'] > 120, 'duration'] = 120


def expand_dataframe(dataframe, window_stride=1):
    dfs = []
    for i, row in dataframe.iterrows():
        for j in range(0, max(1, int(row['duration'])-5+1), window_stride):
            new_fn = row['filename'] + f'_{j}'
            start_end = [j*32000, (j+5)*32000]
            other_cols = [str(val) for val in row.values[1:]]
            dfs.append([new_fn] + start_end + other_cols)
            
        # new_fn = row['filename'] + f'_{0}'
        # start_end = [0*32000, 5*32000]
        # other_cols = [str(val) for val in row.values[1:]]
        # dfs.append([new_fn] + start_end + other_cols)
        
    columns = ['filename', 'start', 'end', 'path', 'duration', 'primary_label', 'secondary_labels']
    dataframe = pd.DataFrame(dfs, columns=columns)
    return dataframe


def get_dataloader(dataframe, config):
    dataset = TestDataset(dataframe, config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=12,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def load_models(exp_list, folds, final_folder):
    models = []
    models_names = []
    for exp_name in exp_list:
        for fold in folds:
            model_folder = filepaths.models_dir / exp_name
            if os.path.isfile(model_folder / final_folder / f'fold{fold}.parquet'):
                continue
            
            folder = 'models' # if fold != -1 else 'chkp'
            fn = f'fold_{fold}_best.pth'# if fold != -1 else f'fold_{fold}_chkp.pth'
            state_path = model_folder / folder / fn
            
            config = load_config(filepaths.models_dir / exp_name / 'config.yaml')
            config.dataset.train_duration = 5
            config.dataset.valid_duration = 5
            config.dataset.labels = labels
            
            model = TattakaModel(config, pretrained=False)
            
            state = torch.load(state_path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            model.eval()
            model.to(device)
            models.append(model)
            models_names.append(f'{exp_name}_{fold}')
            
    return models, models_names


def make_pseudolabels(dataframe, exp_list, folds=(0,1,2,3), window_stride=1, folder='pseudolabels'):
    test_df = dataframe[['filename', 'filepath', 'duration', 'primary_label', 'secondary_labels']].copy().reset_index(drop=True)
    test_df.rename(columns={'filepath': 'path'}, inplace=True)
    test_df['duration'] = test_df['duration'].astype(int)

    test_df = expand_dataframe(test_df, window_stride=window_stride)
    test_df['duration'] = test_df['duration'].astype(int)
    
    config = load_config(filepaths.models_dir / exp_list[0] / 'config.yaml')
    config.dataset.train_duration = 5
    config.dataset.valid_duration = 5
    config.dataset.labels = labels
    test_dl = get_dataloader(test_df, config)

    models, models_names = load_models(exp_list, folds, folder)
    if len(models) == 0:
        return True

    predictions = {model_name: [] for model_name in models_names}
    for step, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
        batch = batch.to(device)

        for model_name, model in zip(models_names, models):
            with torch.no_grad():
                y_pred, output_frame = model(batch)
            logits = y_pred.detach().to('cpu').numpy()
            predictions[model_name].append(logits)

    for model_name, logits in predictions.items():
        exp_name, fold = model_name.split('_')
        fp = filepaths.models_dir / exp_name / folder / f'fold{fold}.parquet'

        preds = np.round(np.concatenate(logits), 3)
        test_df[config.dataset.labels] = preds

        if not os.path.isdir(fp.parent):
            os.makedirs(fp.parent)

        try:
            test_df.drop(['path', 'duration', 'primary_label', 'secondary_labels'], axis=1, inplace=True)
        except:
            pass

        test_df.to_parquet(fp, index=False)
    return True


def concatenate_actual_pseudolabels(exp):
    fp = filepaths.models_dir / exp / 'pseudolabels' / f'pseudolabels.parquet'
    if os.path.isfile(fp):
        return True
    pseudolabels = []
    for fold in tqdm(range(4)):
        pseudo = pd.read_parquet(filepaths.models_dir / exp / 'pseudolabels' / f'fold{fold}.parquet')
        pseudolabels.append(pseudo)
    pseudolabels = pd.concat(pseudolabels, axis=0)
    pseudolabels.to_parquet(fp, index=False)
    return True


def make_mean_pseudolabels(exp, folder):
    new_fp = filepaths.models_dir / exp / folder / f'pseudolabels.parquet'
    pseudolabels = None
    for fold in tqdm(range(4)):
        fp = filepaths.models_dir / exp / folder / f'fold{fold}.parquet'
        pseudo = pd.read_parquet(fp)
        if pseudolabels is None:
            pseudolabels = pseudo[config.dataset.labels].values * 0.25
        else:
            pseudolabels += pseudo[config.dataset.labels].values * 0.25
    pseudo[config.dataset.labels] = pseudolabels
    pseudo.to_parquet(new_fp, index=False)
    return True


def get_exp_list(weights):
    exp_list = sorted(weights.keys())
    return exp_list


def make_weighted_pseudolabels(weights, folds=(0, 1, 2, 3), src_folder='pseudolabels'):
    exp_list = get_exp_list(weights)
    folder_name = '_'.join(sorted(exp_list))
    folder_path = filepaths.models_dir / folder_name / src_folder
    
    for fold in folds:
        pseudolabels = None
        for exp, weight in tqdm(weights.items()):
            pseudo = pd.read_parquet(filepaths.models_dir / exp / src_folder / f'fold{fold}.parquet')
            pseudo = pseudo[pseudo['filename'].isin(pseudo_temp.filename.values)]
            pseudo.sort_values('filename', inplace=True)
            try:
                if pseudolabels is None:
                    pseudolabels = pseudo[config.dataset.labels].values * weight
                else:
                    pseudolabels += pseudo[config.dataset.labels].values * weight
            except:
                print(exp)

        df = pseudo.copy()
        df[config.dataset.labels] = pseudolabels
        
        if 'africa' in src_folder:
            df['filename'] = df['filename'].apply(lambda x: x.split('_')[0])
            df = pd.merge(df, xc_data, on='filename', how='left')
            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
            df.drop(['latitude', 'longitude'], axis=1, inplace=True)
            df['filepath'] = df['filepath'].astype(str)

        fp = folder_path / f'fold{fold}.parquet'
        if not os.path.isdir(fp.parent):
            os.makedirs(fp.parent)
        
        df.to_parquet(fp, index=False)
        
    with open(folder_path / 'weights.json', "w") as outfile:
        json.dump(weights, outfile)
        
    return True


exp_list = ['exp114', 'exp145', 'exp148', 'exp150', 'exp154', 'exp160', 'exp161', 'exp162', 'exp163', 'exp164', 'exp165', ]

for fold in [0, 1, 2, 3]:
    _ = make_pseudolabels(
            dataframe=train[train['fold'] == fold],
            exp_list=exp_list,
            folds=(fold, ),
            window_stride=1,
            folder='pseudolabels' 
        )

weights = {exp: 1/len(exp_list) for exp in exp_list}
make_weighted_pseudolabels(weights, folds=(0, 1, 2, 3), src_folder='pseudolabels')
concatenate_actual_pseudolabels('exp114_exp145_exp148_exp150_exp154_exp160_exp161_exp162_exp163_exp164_exp165')

subset = xc_2023_data.copy()

accepted_licenses = [
    # '//creativecommons.org/licenses/by-nc-nd/4.0/',
    '//creativecommons.org/licenses/by-nc-sa/4.0/',
    # '//creativecommons.org/licenses/by-nc-nd/2.5/',
    # '//creativecommons.org/licenses/by-nc-nd/3.0/',
    '//creativecommons.org/licenses/by-nc-sa/3.0/',
    # '//creativecommons.org/publicdomain/zero/1.0/',
    '//creativecommons.org/licenses/by-sa/4.0/',
    '//creativecommons.org/licenses/by/4.0/',
    '//creativecommons.org/licenses/by-nc/4.0/',
    '//creativecommons.org/licenses/by-sa/3.0/',
]
subset = subset[subset['license'].isin(accepted_licenses)]

exp_list = ['exp114', 'exp145', 'exp148', 'exp150', 'exp154', 'exp160', 'exp161', 'exp162', 'exp163', 'exp164', 'exp165', ]

for fold in [0, 1, 2, 3]:
    _ = make_pseudolabels(
            dataframe=subset,
            exp_list=exp_list,
            folds=(fold, ),
            window_stride=1,
            folder='pseudolabels_xc' # only sa licenses
        )
    
weights = {exp: 1/len(exp_list) for exp in exp_list}
make_weighted_pseudolabels(weights, folds=(0, 1, 2, 3), src_folder='pseudolabels_xc')
make_mean_pseudolabels('exp114_exp145_exp148_exp150_exp154_exp160_exp161_exp162_exp163_exp164_exp165', 'pseudolabels_xc')

