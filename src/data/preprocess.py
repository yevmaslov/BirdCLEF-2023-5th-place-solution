import sys
sys.path.append('../')

import pandas as pd
import numpy as np

from src.environment.utils import load_filepaths

def main():
    filepaths = load_filepaths('../filepaths.yaml')
    train = pd.read_csv(filepaths.raw_dir / 'train.csv')
    test = pd.read_csv(filepaths.raw_dir / 'test.csv')
    submission = pd.read_csv(filepaths.raw_dir / 'sample_submission.csv')

    train.to_parquet(filepaths.processed_dir / 'train.parquet')
    test.to_parquet(filepaths.processed_dir / 'test.parquet')
    submission.to_parquet(filepaths.processed_dir / 'submission.parquet')
    return None

if __name__ == '__main__':
    main()
