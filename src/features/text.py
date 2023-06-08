import pickle
from tqdm.auto import tqdm
import os


def make_text(dataframe):
    dataframe['input_text'] = dataframe['text']
    return dataframe


def tokenize(ids, texts, config):
    tokenized = {}
    for id, text in tqdm(zip(ids, texts), total=len(ids)):
        tok = config.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=config.dataset.max_length,
            truncation=True,
        )
        tokenized[id] = list(tok['input_ids'])
    return tokenized


def get_tokenized_text(dataframe, config, filepaths, load_from_file=True):
    dir_path = filepaths.interim_dir / config.model.backbone_type.replace('/', '-')
    if load_from_file:
        with open(dir_path / 'tokenized_text.pickle', 'rb') as f:
            tokenized_text = pickle.load(f)
    else:
        dataframe = make_text(dataframe)
        tokenized_text = tokenize(dataframe.id.values, dataframe.input_text.values, config)

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        with open(dir_path / 'tokenized_text.pickle', 'wb') as f:
            pickle.dump(tokenized_text, f)

    return tokenized_text
