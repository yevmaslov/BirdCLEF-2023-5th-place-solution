
import pandas as pd
from data.utils import filter_data, upsample_data, update_max_duration, downsample_data
from data.utils import preprocess_url
from data.make_folds import make_folds


def load_dataframe(dataframe_config, filepaths, fold, debug=False, year=2023, remake_folds=False, seed=42, finetune_head=False):
    print(f'Loading {year} data...')

    df = pd.read_csv(filepaths.processed_dir / f'birdclef-{year}' / 'train_metadata.csv')
    folds = pd.read_csv(filepaths.processed_dir / f'birdclef-{year}' / 'folds.csv')
    duration = pd.read_csv(filepaths.processed_dir / f'birdclef-{year}' / 'duration.csv')

    df = pd.merge(df, folds, on='url', how='left')
    df = pd.merge(df, duration, on='url', how='left')
        
    if remake_folds:
        df.drop('fold', axis=1, inplace=True)
        print('Recreating folds with seed: ', seed)
        df = make_folds(df, n_splits=4, random_state=seed)

    df = filter_data(df, drop=dataframe_config.drop, thr=dataframe_config.filter_threshold)

    if not debug:
        train_folds = df[df['fold'] != fold]
        valid_folds = df[df['fold'] == fold]

        train_folds = upsample_data(train_folds, thr=dataframe_config.upsample_threshold)
        df = pd.concat([train_folds, valid_folds], axis=0, ignore_index=True)
        df = df.reset_index(drop=True)
        
    if finetune_head:
        # df = df[df['fold'] != fold].reset_index(drop=True)
        # df.drop('fold', axis=1, inplace=True)
        # print('Recreating folds with seed: ', seed)
        # df = make_folds(df, n_splits=4, random_state=seed)
        train_folds = df[df['fold'] != fold].reset_index(drop=True)
        head_folds = make_folds(train_folds, n_splits=4, random_state=seed)
        head_folds.rename(columns={'fold': 'head_fold'}, inplace=True)
        df = pd.merge(df, head_folds[['filename', 'head_fold']], on='filename', how='left')

    audio_dir = getattr(filepaths, f'birdclef_{year}_audio_dir')
    if year == 2021:
        df['filepath'] = audio_dir / df['primary_label'] / df['filename']
    else:
        df['filepath'] = audio_dir / df['filename']

    # df = update_max_duration(df, dataframe_config.max_duration)
    df['subset'] = f'train_{year}'
    df['duration_float'] = df['duration']
    return df


def load_esc50_noise(filepaths):
    esc50_df = pd.read_csv(filepaths.processed_dir / 'esc50.csv')
    esc50_df['filepath'] = filepaths.esc50_noise_dir / esc50_df['filename']

    selected_noise_categories = [
        'dog', 'chirping_birds', 'thunderstorm',
        'door_wood_knock', 'crow',
        'airplane', 'pouring_water', 'train',
        'sheep', 'water_drops', 'church_bells',
        'wind', 'footsteps', 'frog', 'cow',
        'crackling_fire', 'helicopter',
        'rain', 'insects',
        'breathing', 'coughing',
        'snoring', 'pig',
        'sea_waves', 'cat',
    ]

    esc50_df = esc50_df[esc50_df['category'].isin(selected_noise_categories)]
    esc50_df = esc50_df.reset_index(drop=True)
    return esc50_df


def load_no_call_noise(filepaths):
    no_call_df = pd.read_csv(filepaths.processed_dir / 'no_call.csv')
    no_call_df['filepath'] = filepaths.no_call_noise_audio_dir / no_call_df['filename']
    return no_call_df


def load_zenodo_data(filepaths):
    full_zenodo_data = pd.read_csv(filepaths.processed_dir / 'birdclef-zenodo' / 'annotations.csv')

    full_zenodo_data['filepath'] = filepaths.zenodo_audio_dir / full_zenodo_data['Filename']
    full_zenodo_data['rating'] = 5
    full_zenodo_data['subset'] = 'zenodo'
    full_zenodo_data['fold'] = 99
    full_zenodo_data['duration'] = 5
    
    zenodo_data = full_zenodo_data.copy()
    zenodo_data['fn'] = full_zenodo_data['Filename'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    zenodo_data = zenodo_data.drop_duplicates(subset=['fn']).reset_index(drop=True)
    return full_zenodo_data, zenodo_data


def load_xc_data(dataframe_config, filepaths, year=2023):
    if year == 2023:
        fn = 'metadata_full.csv' if dataframe_config.full else 'metadata_sa.csv'
    else:
        fn = 'metadata_full.csv'
    xc_data = pd.read_csv(filepaths.processed_dir / f'birdclef-{year}_XenoCanto' / fn)
    xc_data['filename'] = xc_data['filename'].str.replace('.mp3', '.wav')

    audio_dir = getattr(filepaths, f'xeno_canto_{year}_audio_dir')
    xc_data['filepath'] = audio_dir / xc_data['filename']
    xc_data['rating'].fillna(1, inplace=True)

    if 'duration' in xc_data.columns:
        xc_data.drop(columns=['duration'], inplace=True)

    duration = pd.read_csv(filepaths.processed_dir / f'birdclef-{year}_XenoCanto' / 'duration.csv')
    xc_data['url'] = xc_data['url'].apply(preprocess_url)
    duration['url'] = duration['url'].apply(preprocess_url)

    xc_data = pd.merge(xc_data, duration, on='url', how='left')
    xc_data = xc_data[xc_data['duration'] >= 1].reset_index(drop=True)

    xc_data['fold'] = 99
    # xc_data.loc[xc_data['duration'] > 120, 'duration'] = 120
    # xc_data = update_max_duration(xc_data, dataframe_config.max_duration)
    
    xc_data_other = xc_data[xc_data['primary_label'].isin(['other', 'africa'])]
    xc_data_rest = xc_data[~xc_data['primary_label'].isin(['other', 'africa'])]
    
    xc_data_rest = downsample_data(xc_data_rest, thr=dataframe_config.downsample_threshold)
    xc_data = pd.concat([xc_data_rest, xc_data_other], ignore_index=True)
    xc_data = xc_data.reset_index(drop=True)
    xc_data['duration_float'] = xc_data['duration']
    
    print(f'XenoCanto {year} shape: ', xc_data.shape)
    return xc_data
