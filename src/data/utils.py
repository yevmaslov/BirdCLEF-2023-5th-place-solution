import pandas as pd
import numpy as np
import soundfile as sf
import sys
sys.path.append('../src')
from data.labels import get_2020_labels, get_2021_labels, get_2023_labels, get_2022_labels


def filter_data(df, drop=False, thr=5):
    counts = df.primary_label.value_counts()
    if drop:
        df = df[df['primary_label'].isin(counts[counts > thr].index)].reset_index(drop=True)
    else:
        df.loc[df['primary_label'].isin(counts[counts < thr].index), 'fold'] = 99
    return df


def upsample_data(df, thr=10):
    class_dist = df['primary_label'].value_counts()
    down_classes = class_dist[class_dist < thr].index.tolist()

    up_dfs = []

    for c in down_classes:
        class_df = df.query("primary_label==@c")
        num_up = thr - class_df.shape[0]
        class_df = class_df.sample(n=num_up, replace=True, random_state=42)
        up_dfs.append(class_df)

    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    return up_df


def upsample_xeno_canto(train_df, xeno_canto_df, config, year=2023):
    xc_data = xeno_canto_df.copy()

    counts = train_df.primary_label.value_counts()

    _config = getattr(config.dataframe, f'xc_{year}_data_config')
    upsample_smallest_n = _config.upsample_smallest_n
    counts = counts[counts < upsample_smallest_n]

    small_classes = xc_data[xc_data.primary_label.isin(counts.index.values)].copy()
    rest_classes = xc_data[~xc_data.primary_label.isin(counts.index.values)].copy()

    small_classes = upsample_data(small_classes, config.dataframe.xc_2023_data_config.upsample_threshold)

    xc_data = pd.concat([small_classes, rest_classes], ignore_index=True)
    xc_data = xc_data.reset_index(drop=True)
    return xc_data


def downsample_data(df, thr=1000):
    down_dfs = []
    for label in df.primary_label.unique():
        df_label = df[df.primary_label == label]
        if df_label.shape[0] > thr:
            df_label = df_label.sort_values('rating', ascending=False)
            df_label = df_label.head(thr)
        
        down_dfs.append(df_label)
    df = pd.concat(down_dfs, axis=0, ignore_index=True).reset_index(drop=True)
    return df


def make_weight(df, from_rating=True):
    if from_rating:
        df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
    else:
        df['weight'] = 1
    return df


def downsample_data(df, thr=1000):
    down_dfs = []
    for label in df.primary_label.unique():
        df_label = df[df.primary_label == label]
        if df_label.shape[0] > thr:
            df_label = df_label.sort_values('rating', ascending=False)
            df_label = df_label.head(thr)
        
        down_dfs.append(df_label)
    df = pd.concat(down_dfs, axis=0, ignore_index=True).reset_index(drop=True)
    return df


def get_labels(config):
    labels = []

    if config.dataframe.train_2023_config.use or config.dataframe.xc_2023_data_config.use or config.dataframe.pseudo_2023_config.use or config.dataframe.pseudo_2023_xc_config.use:
        labels.extend(get_2023_labels())
    if config.dataframe.train_2022_config.use or config.dataframe.xc_2022_data_config.use or config.dataframe.pseudo_2022_config.use or config.dataframe.pseudo_2022_xc_config.use:
        labels.extend(get_2022_labels())
    if config.dataframe.train_2021_config.use or config.dataframe.xc_2021_data_config.use or config.dataframe.pseudo_2021_config.use or config.dataframe.pseudo_2021_xc_config.use:
        labels.extend(get_2021_labels())
    # if config.dataframe.train_2020_config.use or config.dataframe.xc_2020_data_config.use or config.dataframe.pseudo_2020_config.use or config.dataframe.pseudo_2020_xc_config.use:
    #     labels.extend(get_2020_labels())
    
    labels = sorted([lab for lab in labels if lab not in ['other', 'africa']])
    return labels


def make_label_columns(df, labels):
    for label in labels:
        primary = df['primary_label'] == label
        secondary = df['secondary_labels'].apply(lambda x: label in x)
        df[label] = (primary | secondary).astype(int)
    return df


def update_max_duration(df, max_duration=180):
    df['md'] = max_duration
    df['duration'] = df[['duration', 'md']].min(axis=1)
    df = df.drop('md', axis=1)
    return df


def preprocess_url(url):
    url = url.replace('www.', '')
    url = url.replace('http://', 'https://')
    return url


def get_duration(fp):
    wav, sr = sf.read(fp)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    return len(wav) / sr
