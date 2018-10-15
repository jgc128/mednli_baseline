import json

from utils.pickle import load_pickle, save_pickle


def get_tokens(sentence_binary_parse):
    sentence = sentence_binary_parse \
        .replace('(', ' ').replace(')', ' ') \
        .replace('-LRB-', '(').replace('-RRB-', ')') \
        .replace('-LSB-', '[').replace('-RSB-', ']')

    tokens = sentence.split()

    return tokens


def read_mednli(filename):
    data = []

    with open(filename, 'r') as f:
        for line in f:
            example = json.loads(line)

            premise = get_tokens(example['sentence1_binary_parse'])
            hypothesis = get_tokens(example['sentence2_binary_parse'])
            label = example['gold_label']

            data.append((premise, hypothesis, label))

    print(f'MedNLI file loaded: {filename}, {len(data)} examples')
    return data


def load_mednli(cfg):
    filenames = [
        'mli_train_v1.jsonl',
        'mli_dev_v1.jsonl',
        'mli_test_v1.jsonl',
    ]
    filenames = [cfg.data_dir.joinpath(f) for f in filenames]

    mednli_train, mednli_dev, mednli_test = [read_mednli(f) for f in filenames]

    return mednli_train, mednli_dev, mednli_test
