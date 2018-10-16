import json

from nltk import word_tokenize


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
            label = example.get('gold_label', None)

            data.append((premise, hypothesis, label))

    print(f'MedNLI file loaded: {filename}, {len(data)} examples')
    return data


def read_sentences(filename):
    with open(filename, 'r') as f:
        lines = [l.split('\t') for l in f.readlines()]

    input_data = [(word_tokenize(l[0]), word_tokenize(l[1]), None) for l in lines if len(l) == 2]
    return input_data


def load_mednli(cfg):
    filenames = [
        'mli_train_v1.jsonl',
        'mli_dev_v1.jsonl',
        'mli_test_v1.jsonl',
    ]
    filenames = [cfg.mednli_dir.joinpath(f) for f in filenames]

    mednli_train, mednli_dev, mednli_test = [read_mednli(f) for f in filenames]

    return mednli_train, mednli_dev, mednli_test
