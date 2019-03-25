import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from utils.dataset import NLIDataset
from utils.helpers import create_model
from utils.mednli import read_mednli, read_sentences
from utils.pickle import load_pickle
from utils.torch import load_weights, create_data_loader, to_device


def save_predictions(predictions, filename):
    with open(filename, 'w') as f:
        labels = sorted(NLIDataset.LABEL_TO_ID.keys(), key=lambda l: NLIDataset.LABEL_TO_ID[l])
        f.write(','.join(labels) + '\n')
        np.savetxt(f, predictions, fmt='%.5f', delimiter=',')

    print(f'Saved: {filename}')


def get_input_data(filename):
    input_data = None

    if filename.suffix == '.jsonl':
        input_data = read_mednli(filename)

    if filename.suffix == '.txt':
        input_data = read_sentences(filename)

    if input_data is None:
        raise ValueError(f'Cannot determine input file format: {filename}')

    return input_data


def main(model_spec_filename, input_filename, output_filename):
    model_spec = load_pickle(model_spec_filename)
    model_name = model_spec['model_name']
    model_params = model_spec['model_params']
    vocab = model_spec['vocab']
    cfg = model_spec['cfg']

    print(f'Model name: {model_name}')
    print(f'Model params: {model_params}')

    model = create_model(cfg, model_params)
    model.eval()
    load_weights(model, cfg.models_dir.joinpath(f'{model_name}.pt'))

    input_data = get_input_data(input_filename)
    print(f'Input data: {len(input_data)}')

    dataset = NLIDataset(input_data, vocab=vocab, lowercase=cfg.lowercase, max_len=cfg.max_len)
    data_loader = create_data_loader(dataset, cfg.batch_size, shuffle=False)

    predictions = []
    for batch in data_loader:
        (premise, hypothesis), _ = to_device(batch)
        logits = model(premise, hypothesis)
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        predictions.append(probabilities)

    predictions = np.concatenate(predictions)
    print(f'Predictions: {predictions.shape}')

    save_predictions(predictions, output_filename)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: {__file__} <model_spec_filename> <input_file> <output_file>')
        sys.exit(1)

    model_spec_filename = Path(sys.argv[1])
    input_filename = Path(sys.argv[2])
    output_filename = Path(sys.argv[3])
    main(model_spec_filename, input_filename, output_filename)
