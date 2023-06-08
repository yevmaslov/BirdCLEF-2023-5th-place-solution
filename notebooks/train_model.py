
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import os
import shutil
import sys

from copy import deepcopy

sys.path.append('../src')
sys.path.append('birdcleff-2023/src')

from data.load_data import load_esc50_noise, load_no_call_noise, load_dataframe, load_zenodo_data, load_xc_data

from environment.utils import load_filepaths, load_config, seed_everything, create_run_folder, save_config, str2bool
from logger.wandb import init_wandb
from models.dataset import CustomDataset
from models.model import get_model, load_state
from models.trainer import Trainer

from models.utils import get_valid_steps
from models.optimizer import get_optimizer
from models.scheduler import get_scheduler
from logger.logger import Logger
from loss import padded_cmap

from tqdm.notebook import tqdm
from data.utils import make_weight, get_labels, make_label_columns, downsample_data, upsample_data, upsample_xeno_canto
from types import SimpleNamespace

tqdm.pandas()
import warnings

from argparse import ArgumentParser
import soundfile as sf
from pathlib import Path

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")

def process_secondary_labels(labels):
    labels = labels.replace('[', '')
    labels = labels.replace(']', '')
    labels = labels.replace("'", '')
    labels = labels.split(',')
    labels = [lab.strip() for lab in labels]
    labels = [lab for lab in labels if lab in config.dataset.labels]
    return labels


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=True
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0
    )
    parser.add_argument(
        '--state_from_fold',
        type=int,
        default=-99
    )
    parser.add_argument(
        '--remake_folds',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '--exp_name',
        type=str,
    )
    parser.add_argument(
        '--job_type',
        type=str,
        default='train'
    )
    parser.add_argument(
        '--use_2021_data',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=None
    )
    args = parser.parse_args()
    return args


def get_default_args():
    args = SimpleNamespace()
    args.debug = True
    args.fold = 0
    args.state_from_fold = 0
    args.remake_folds = False
    args.exp_name = 'test'
    args.job_type = 'train'
    args.use_2021_data = False
    return args


# args = get_default_args()
args = parse_args()
assert args.job_type in ['train', 'pretrain',]

job_type_to_config = {
    'train': '../finetune_config.yaml',
    'pretrain': '../pretrain_config.yaml',
}

if args.config_path is None:
    print('Config from: ', job_type_to_config[args.job_type])
    config = load_config(job_type_to_config[args.job_type])
else:
    print('Config from: ', args.config_path)
    config = load_config(args.config_path)

filepaths = load_filepaths('../filepaths.yaml')

debug = args.debug
pretrained = True

config.experiment_name = args.exp_name
if args.job_type == 'pretrain':
    config.experiment_name = config.experiment_name + '_pretrain'
elif args.job_type in ['train_1', 'train_2']:
    n = args.job_type.split('_')[1]
    config.experiment_name += f'_{n}'

config.fold = args.fold

if debug:
    config.debug = True
    config.experiment_name = 'test'
    config.logger.use_wandb = False

    config.dataset.train_batch_size = 2
    config.dataset.valid_batch_size = 2

config.run_name = config.experiment_name + f'_fold{config.fold}'
config.run_id = config.run_name + "_1" + f'_{args.state_from_fold}'
run = init_wandb(config)

config.dataset.valid_duration = config.dataset.train_duration
config.dataset.train_audio_len = config.dataset.train_duration * config.dataset.sample_rate
config.dataset.valid_audio_len = config.dataset.valid_duration * config.dataset.sample_rate

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(config.environment.seed)

filepaths.run_dir = filepaths.models_dir / config.experiment_name
if debug and os.path.isdir(filepaths.run_dir):
    shutil.rmtree(filepaths.run_dir)
if not os.path.isdir(filepaths.run_dir):
    create_run_folder(filepaths.run_dir)

save_config(config, path=filepaths.run_dir / 'config.yaml')

labels = sorted(get_labels(config))
config.dataset.labels = labels
print(config.environment.seed)
train23 = load_dataframe(config.dataframe.train_2023_config, filepaths, config.fold, debug=debug, year=2023, remake_folds=args.remake_folds, seed=config.environment.seed)
train22 = load_dataframe(config.dataframe.train_2022_config, filepaths, config.fold, debug=debug, year=2022)
train21 = load_dataframe(config.dataframe.train_2021_config, filepaths, config.fold, debug=debug, year=2021)


# full_zenodo_data, zenodo_data = load_zenodo_data(filepaths)

xc_2023_data = load_xc_data(config.dataframe.xc_2023_data_config, filepaths, year=2023)
xc_2023_data = upsample_xeno_canto(train23, xc_2023_data, config, year=2023)

xc_2022_data = load_xc_data(config.dataframe.xc_2022_data_config, filepaths, year=2022)

esc50_df = load_esc50_noise(filepaths)
no_call_df = load_no_call_noise(filepaths)

train = []


if config.dataframe.pseudo_2023_config.use:
    pseudo_fn = 'pseudolabels.parquet' if args.fold == -1 else f'fold{args.fold}.parquet'
    if args.exp_name in ['exp122', 'exp168',]:
        pseudo_fn = 'pseudolabels.parquet'
    fp = filepaths.models_dir / config.dataframe.pseudo_2023_config.path / pseudo_fn
    print('Train Pseudo from: ', fp)
    
    full_train23_pseudolabels = pd.read_parquet(fp)
    full_train23_pseudolabels['filename'] = full_train23_pseudolabels['filename'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    
    if 'duration' not in full_train23_pseudolabels.columns:
        full_train23_pseudolabels = pd.merge(full_train23_pseudolabels, train23[['filename', 'duration', 'primary_label', 'secondary_labels']], on='filename', how='left')
        
    full_train23_pseudolabels['filepath'] = filepaths.birdclef_2023_audio_dir / full_train23_pseudolabels['filename']
    full_train23_pseudolabels = pd.merge(full_train23_pseudolabels, train23[['filename', 'fold', 'rating',]], on='filename', how='left')
    full_train23_pseudolabels = make_weight(full_train23_pseudolabels, from_rating=config.dataframe.weight_from_rating)
    full_train23_pseudolabels['duration_float'] = full_train23_pseudolabels['duration']
    train23_pseudolabels = full_train23_pseudolabels.copy()
    train23_pseudolabels = train23_pseudolabels.drop_duplicates(subset=['filename']).reset_index(drop=True)
    train23_pseudolabels['subset'] = 'pseudo'
    train23_pseudolabels = train23_pseudolabels[train23_pseudolabels['fold'] != config.fold].reset_index(drop=True)
    
    full_train23_pseudo_dict = {}
    for fn in train23_pseudolabels['filename'].values:
        full_train23_pseudo_dict[fn] = full_train23_pseudolabels[full_train23_pseudolabels['filename'] == fn].reset_index(drop=True)
    config.full_train23_pseudo = full_train23_pseudo_dict

    train.append(train23_pseudolabels)
    train23 = train23[~train23['filename'].isin(train23_pseudolabels['filename'].values)].reset_index(drop=True)
    
    

if config.dataframe.pseudo_2023_xc_config.use:
    pseudo_fn = 'pseudolabels.parquet' if args.fold == -1 else f'fold{args.fold}.parquet'
    if args.exp_name in ['exp122', 'exp168',]:
        pseudo_fn = 'pseudolabels.parquet'
    fp = filepaths.models_dir / config.dataframe.pseudo_2023_xc_config.path / pseudo_fn
    print('XC Pseudo from: ', fp)
    full_train23_pseudolabels_xc = pd.read_parquet(fp)
    full_train23_pseudolabels_xc['filename'] = full_train23_pseudolabels_xc['filename'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    
    if 'duration' not in full_train23_pseudolabels_xc.columns:
        full_train23_pseudolabels_xc = pd.merge(full_train23_pseudolabels_xc, xc_2023_data[['filename', 'duration', 'primary_label', 'secondary_labels']], on='filename', how='left')
        
    full_train23_pseudolabels_xc['filepath'] = filepaths.xeno_canto_2023_audio_dir / full_train23_pseudolabels_xc['filename']
    full_train23_pseudolabels_xc = pd.merge(full_train23_pseudolabels_xc, xc_2023_data[['filename', 'rating',]], on='filename', how='left')
    full_train23_pseudolabels_xc = make_weight(full_train23_pseudolabels_xc, from_rating=config.dataframe.weight_from_rating)
    full_train23_pseudolabels_xc['duration_float'] = full_train23_pseudolabels_xc['duration']
    print('2023 xc pseudolabels shape: ', full_train23_pseudolabels_xc.shape)

    train23_pseudolabels_xc = full_train23_pseudolabels_xc.copy()
    train23_pseudolabels_xc = train23_pseudolabels_xc.drop_duplicates(subset=['filename']).reset_index(drop=True)
    train23_pseudolabels_xc['subset'] = 'pseudo_xc'
    
    full_train23_pseudo_xc_dict = {}
    for fn in train23_pseudolabels_xc['filename'].values:
        full_train23_pseudo_xc_dict[fn] = full_train23_pseudolabels_xc[full_train23_pseudolabels_xc['filename'] == fn].reset_index(drop=True)
    config.full_train23_xc_pseudo = full_train23_pseudo_xc_dict

    train23_pseudolabels_xc = train23_pseudolabels_xc[train23_pseudolabels_xc['duration_float'].notna()].reset_index(drop=True)
    print('2023 xc pseudolabels shape: ', train23_pseudolabels_xc.shape)
    train.append(train23_pseudolabels_xc)
    config.dataframe.xc_2023_data_config.use = False

    
if config.dataframe.train_2023_config.use:
    train.append(train23)
else:
    if args.job_type != 'pretrain':
        train.append(train23[train23['fold'] == args.fold])
    

if config.dataframe.train_2022_config.use:
    train.append(train22)
if config.dataframe.train_2021_config.use:
    train.append(train21)
# if config.dataframe.zendodo_config.use:
#     train.append(zenodo_data)
if config.dataframe.xc_2023_data_config.use:
    train.append(xc_2023_data)
if config.dataframe.xc_2022_data_config.use:
    train.append(xc_2022_data)


train = pd.concat(train, ignore_index=True)
train = train.reset_index(drop=True)
train['rating'] = train['rating'].replace(0, 1)

print('Duration float is na: ', train[train['duration_float'].isna()].shape)
train = train[train['duration_float'].notna()].reset_index(drop=True)

train['duration_float'] = train['duration_float'].astype(int)

if hasattr(config.dataframe, 'drop_other') and config.dataframe.drop_other:
    train = train[train['primary_label'] != 'other'].reset_index(drop=True)


if config.dataframe.pseudo_2023_xc_config.use or config.dataframe.xc_2023_data_config.use:
    other_xc = train[train['primary_label'] == 'other'].copy()
    other_xc['secondary_labels'] = other_xc['secondary_labels'].apply(process_secondary_labels)
    other_xc['labels_len'] = other_xc['secondary_labels'].apply(lambda x: len(x))
    other_xc = other_xc[other_xc['labels_len'] > 0]
    mask = other_xc['labels_len'] == 1
    other_xc['first_secondary_label'] = other_xc['secondary_labels'].apply(lambda x: x[0])
    other_xc.loc[mask, 'primary_label'] = other_xc.loc[mask, 'first_secondary_label']
    other_other = other_xc[other_xc['primary_label'] == 'other'].copy()
    other_rest = other_xc[other_xc['primary_label'] != 'other'].copy()
    
    if other_rest.shape[0] > 0 and args.exp_name not in ['exp122']:
        other_rest = downsample_data(other_rest, thr=1000)
    
    other_xc = pd.concat([other_rest, other_other], ignore_index=True)
    other_xc.to_csv('other_test.csv', index=False)
    other_xc = pd.read_csv('other_test.csv')
    other_xc['filepath'] = other_xc['filepath'].apply(lambda x: Path(x))
    other_xc.drop(['labels_len', 'first_secondary_label'], axis=1, inplace=True)
    other_xc['primary_label'] = 'other'
    train = pd.concat([train[train['primary_label'] != 'other'], other_xc], ignore_index=True)
    train = train.reset_index(drop=True)


print('Full train shape:', train.shape)
print('Duration is nan count: ', train['duration_float'].isna().sum())
train = train[train['duration_float'].notna()].reset_index(drop=True)

config.dataset.esc50_df = esc50_df
config.dataset.no_call_df = no_call_df

train = make_weight(train, from_rating=config.dataframe.weight_from_rating)
train = make_label_columns(train, labels)


# full_zenodo_data = make_label_columns(full_zenodo_data, labels)
# full_zenodo_data = make_weight(full_zenodo_data, from_rating=config.dataframe.weight_from_rating)
# full_zenodo_data_dict = {}
# for fn in zenodo_data['fn'].values:
#     full_zenodo_data_dict[fn] = full_zenodo_data[full_zenodo_data['fn'] == fn].reset_index(drop=True)
# config.full_zenodo_data = full_zenodo_data_dict

print('Shape of labels: ', train[config.dataset.labels].shape)

if config.fold != -1:
    train_folds = train[train['fold'] != config.fold].copy().reset_index(drop=True)
    valid_folds = train[train['fold'] == config.fold].copy().reset_index(drop=True)
else:
    train_folds = train.copy().reset_index(drop=True)
    valid_folds = train.sample(1000).copy().reset_index(drop=True)

if debug:
    train_folds = train_folds.sample(1000).reset_index(drop=True)
    valid_folds = valid_folds.sample(100).reset_index(drop=True)


if hasattr(config.dataframe, 'use_africa') and config.dataframe.use_africa:
    pseudo_fn = 'pseudolabels.parquet' if args.fold == -1 else f'fold{args.fold}.parquet'
    fp = f'../models/{config.dataframe.africa_pseudo_path}/{pseudo_fn}'
    print('Africa Pseudo from: ', fp)
    africa = pd.read_parquet(fp)
    print('Africa shape: ', africa.shape)
    
    short_africa = africa.copy()
    short_africa = short_africa.drop_duplicates(subset=['filename']).reset_index(drop=True)
    short_africa['subset'] = 'africa'
    
    full_africa_dict = {}
    for fn in short_africa['filename'].values:
        full_africa_dict[fn] = africa[africa['filename'] == fn].reset_index(drop=True)
    config.full_africa = full_africa_dict
    
    # config.full_africa = africa
    
    train_folds = pd.concat([train_folds, short_africa], ignore_index=True)
    train_folds = train_folds.reset_index()

print('Train folds shape:', train_folds.shape)
print('Valid folds shape:', valid_folds.shape)

config.dataset.train_df = train_folds

mixup_df = train_folds[train_folds['subset'].isin(['train_2023', 'pseudo'])].copy()
counts = mixup_df.primary_label.value_counts()
counts = counts[counts < config.dataframe.mixup_thr]
mixup_df = mixup_df[mixup_df['primary_label'].isin(counts.index.values)].reset_index(drop=True)

print('Mixup shape: ', mixup_df.shape)

config.dataset.mixup_df = mixup_df

not_other = train_folds[~train_folds['primary_label'].isin(['other', 'africa'])].reset_index(drop=True)
print(not_other.shape)
other = train_folds[train_folds['primary_label'].isin(['other', 'africa'])].reset_index(drop=True)
print(other.shape)
not_other = upsample_data(not_other, thr=50)
train_folds = pd.concat([not_other, other], ignore_index=True)
train_folds = train_folds.reset_index(drop=True)
print('Train folds shape:', train_folds.shape)

train_dataset = CustomDataset(train_folds, config, train=True)
valid_dataset = CustomDataset(valid_folds, config, train=False)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.dataset.train_batch_size,
    num_workers=12,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.dataset.valid_batch_size,
    num_workers=12,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

pretrained_models_dict = {
    'tf_efficientnet_b0_ns': 'exp149_pretrain',
    'tf_efficientnet_b1_ns': 'exp168_pretrain',
    'tf_efficientnet_b2_ns': 'exp148_pretrain',
    'tf_efficientnet_b3_ns': 'exp130_pretrain',
    'tf_efficientnet_b4_ns': 'exp164_pretrain',
    'tf_efficientnet_b5_ns': 'exp144_pretrain',
    'tf_efficientnet_b6_ns': 'exp163_pretrain',
    'tf_efficientnet_b7_ns': 'exp165_pretrain',
    'efficientnet_b1': 'exp091_pretrain',
    'tf_efficientnetv2_s_in21k': 'exp157_pretrain',
    'tf_efficientnetv2_m_in21k': 'exp166_pretrain',
    'tf_efficientnetv2_l_in21k': 'exp169_pretrain',
}

fullfit_pretrained_models_dict = {
    'tf_efficientnet_b0_ns': 'exp247_pretrain',
    'tf_efficientnet_b1_ns': 'exp244_pretrain',
    'tf_efficientnet_b2_ns': 'exp231_pretrain',
    'tf_efficientnet_b3_ns': 'exp248_pretrain',
}

def freeze(module): 
    for parameter in module.parameters():
        parameter.requires_grad = False

        
model = get_model(config, pretrained=True)
if pretrained and args.job_type not in ['pretrain', 'train_1']:
    if config.model.backbone_type == 'efficientnet_b1':
        model = load_state(model, config, filepaths)
    else:
        pretrained_folder = pretrained_models_dict[config.model.backbone_type]
        
        # States trained on 3/4 folds of 2021 and 2022 data
        if config.model.backbone_type in ['tf_efficientnetv2_s_in21k', 'tf_efficientnetv2_m_in21k', 'tf_efficientnetv2_l_in21k']:
            print('State from: ', f'../models/{pretrained_folder}/models/fold_{0}_best.pth')
            state = torch.load(f'../models/{pretrained_folder}/models/fold_{0}_best.pth')
        else:
            if config.model.backbone_type == 'tf_efficientnet_b2_ns' and args.fold == -1:
                print('State from: ', f'../models/{pretrained_folder}/models/fold_{0}_best.pth')
                state = torch.load(f'../models/{pretrained_folder}/models/fold_{0}_best.pth')
            else:
                fold = args.state_from_fold if args.state_from_fold != -99 else args.fold
                print('State from: ', f'../models/{pretrained_folder}/models/fold_{fold}_best.pth')
                state = torch.load(f'../models/{pretrained_folder}/models/fold_{fold}_best.pth')
            

        # Fulfit states, trained on 2021, 2022, 2023 actual data
        if args.exp_name in ['exp280', 'exp283', 'exp285', 'exp286', 'exp291', 'exp290', 'exp294', 'exp295', 'exp302']:
            if args.fold == -1 and config.model.backbone_type != 'tf_efficientnet_b4_ns' and config.model.backbone_type != 'tf_efficientnetv2_s_in21k':
                pretrained_folder = fullfit_pretrained_models_dict[config.model.backbone_type]
                state_path =  f'../models/{pretrained_folder}/models/fold_{args.fold}_best.pth'
                print('State from: ', state_path)
                state = torch.load(state_path)
            
        state = {
            key.replace('model.', ''): value for key, value in state['model'].items()
            if key.replace('model.', '') not in ["conv_head.weight", "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked", "pool.p", "clf.weight", "clf.bias"]
        }
        model.model.load_state_dict(state)
            
            
        # Load only head, different number of classes between states
        if config.model.finetune_head:

            # Finetune head, load only body, head reinitialized
            if args.fold == -1:
                print('State from: ', f'../models/{config.model.model_state}/chkp/fold_{args.fold}_chkp.pth')
                state = torch.load(f'../models/{config.model.model_state}/chkp/fold_{args.fold}_chkp.pth')
            else:
                print('State from: ', f'../models/{config.model.model_state}/models/fold_{args.fold}_best.pth')
                state = torch.load(f'../models/{config.model.model_state}/models/fold_{args.fold}_best.pth')
            
            state = {
                key.replace('model.', ''): value for key, value in state['model'].items()
                if key.replace('model.', '') not in ["head.pooling.p", "head.dense_layers.1.weight", "head.dense_layers.1.bias", "head.attention.weight", "head.attention.bias", "head.fix_scale.weight", "head.fix_scale.bias"]
            }
            model.model.load_state_dict(state)
            freeze(model.model)


# # Encoder - classifier models, load first 3 (for b2 and b3) or 4 (b1 and b0) layers - train remaining
# if config.model.finetune_top_layers:
#     config.freeze_n_layers = 3 if config.model.backbone_type in ['tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns'] else 4
#     if config.model.backbone_type in ['tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns']:
#         print('State from (encoder): ', f'../models/{config.model.encoder_state}/chkp/fold_{-1}_chkp.pth')
#         state = torch.load(f'../models/{config.model.encoder_state}/chkp/fold_{-1}_chkp.pth')

#         new_config = deepcopy(config)
#         new_config.model.backbone_type = 'tf_efficientnet_b1_ns'
#         encoder = get_model(new_config, pretrained=True)

#         encoder.load_state_dict(state['model'])

#         model.model.blocks[0] = encoder.model.blocks[0]
#         model.model.blocks[1] = encoder.model.blocks[1]
#         model.model.blocks[2] = encoder.model.blocks[2]
#         model.model.blocks[3] = encoder.model.blocks[3]
#         model.model.conv_stem = encoder.model.conv_stem
#         model.model.bn1 = encoder.model.bn1

#         freeze(model.model.blocks[:config.freeze_n_layers])
#         freeze(model.model.conv_stem)
#         freeze(model.model.bn1)

#         model.model.blocks[:config.freeze_n_layers].eval()
#         model.model.conv_stem.eval()
#         model.model.bn1.eval()


#     elif config.model.backbone_type in ['tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns']:
#         print('State from (encoder): ', f'../models/{config.model.encoder_state}/chkp/fold_{-1}_chkp.pth')
#         state = torch.load(f'../models/{config.model.encoder_state}/chkp/fold_{-1}_chkp.pth')

#         new_config = deepcopy(config)
#         new_config.model.backbone_type = 'tf_efficientnet_b3_ns'
#         encoder = get_model(new_config, pretrained=True)

#         # new_config.model.backbone_type = 'tf_efficientnet_b2_ns'
#         # encoder = get_model(new_config, pretrained=True)
        
#         # encoder.model.blocks[0] = _encoder.model.blocks[0]
#         # encoder.model.blocks[1] = _encoder.model.blocks[1]
#         # encoder.model.blocks[2] = _encoder.model.blocks[2]
#         # # model.model.blocks[3] = encoder.model.blocks[3]
#         # encoder.model.conv_stem = _encoder.model.conv_stem
#         # encoder.model.bn1 = _encoder.model.bn1

#         encoder.load_state_dict(state['model'])

#         model.model.blocks[0] = encoder.model.blocks[0]
#         model.model.blocks[1] = encoder.model.blocks[1]
#         model.model.blocks[2] = encoder.model.blocks[2]
#         # model.model.blocks[3] = encoder.model.blocks[3]
#         model.model.conv_stem = encoder.model.conv_stem
#         model.model.bn1 = encoder.model.bn1

#         freeze(model.model.blocks[:config.freeze_n_layers])
#         freeze(model.model.conv_stem)
#         freeze(model.model.bn1)

#         model.model.blocks[:config.freeze_n_layers].eval()
#         model.model.conv_stem.eval()
#         model.model.bn1.eval()

if config.model.finetune_top_layers:
    config.freeze_n_layers = 3
    if args.exp_name in ['exp290', 'exp302']:
        config.freeze_n_layers = 4
    
    print('State from (encoder): ', f'../models/{config.model.encoder_state}/chkp/fold_{-1}_chkp.pth')
    state = torch.load(f'../models/{config.model.encoder_state}/chkp/fold_{-1}_chkp.pth')

    new_config = load_config(f'../models/{config.model.encoder_state}/config.yaml')
    new_config.dataset.labels = config.dataset.labels
    encoder = get_model(new_config, pretrained=True)

    encoder.load_state_dict(state['model'])

    model.model.blocks[0] = encoder.model.blocks[0]
    model.model.blocks[1] = encoder.model.blocks[1]
    model.model.blocks[2] = encoder.model.blocks[2]
    if args.exp_name in ['exp290', 'exp302']:
        model.model.blocks[3] = encoder.model.blocks[3]
    model.model.conv_stem = encoder.model.conv_stem
    model.model.bn1 = encoder.model.bn1

    freeze(model.model.blocks[:config.freeze_n_layers])
    freeze(model.model.conv_stem)
    freeze(model.model.bn1)

    model.model.blocks[:config.freeze_n_layers].eval()
    model.model.conv_stem.eval()
    model.model.bn1.eval()

    
print(config.model.backbone_type)
model.to(device)

train_steps_per_epoch = int(len(train_dataloader))
num_train_steps = train_steps_per_epoch * config.training.epochs
eval_steps = get_valid_steps(int(len(train_dataloader)), config.training.evaluate_n_times_per_epoch)

optimizer = get_optimizer(model, config)
scheduler = get_scheduler(optimizer, config, num_train_steps)

logger = Logger(
    train_steps=len(train_dataloader),
    valid_steps=len(valid_dataloader),
    config=config,
    eval_steps=eval_steps,
    output_file=filepaths.run_dir / 'logs' / f'fold-{config.fold}.log'
)


def compute_score(model, valid_losses, predictions):
    predictions = torch.tensor(predictions)
    predictions = predictions.cpu().detach().numpy()

    preds = valid_folds[labels].copy()
    preds[labels] = predictions
    score = padded_cmap(valid_folds[labels], preds[labels], padding_factor=5)
    return score



trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    logger=logger,
    config=config,
    compute_score_fn=compute_score,
    output_dir=filepaths.run_dir,
    eval_steps=eval_steps,
    direction='maximize',
)

trainer.train()
