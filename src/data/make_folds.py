from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
import pandas as pd
import sys
sys.path.append('../')
from src.environment.utils import load_filepaths


def make_folds(dataframe, n_splits, random_state):
    df = dataframe.copy()
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for n, (_, val_index) in enumerate(k_fold.split(df, df['primary_label'])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df


if __name__ == '__main__':
    filepaths = load_filepaths('../filepaths.yaml')

    train = pd.read_csv(filepaths.raw_dir / 'train_metadata.csv')
    train = make_folds(train, n_splits=4, random_state=42)
    train = train[['url', 'fold']]
    train.to_csv(filepaths.processed_dir / 'folds.csv', index=False)
