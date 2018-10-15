import numpy as np

import models
from config import Models, WordEmbeddings
from utils.dataset import NLIDataset
from utils.mednli import load_mednli
from utils.pickle import load_pickle, save_pickle
from utils.torch import init_weights, to_device
from utils.vocab import Vocab


def get_model_params(cfg, W_emb):
    model_params = dict(
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        trainable_embeddings=cfg.trainable_embeddings,

        vocab_size=W_emb.shape[0],
        embedding_size=W_emb.shape[1],
    )

    return model_params


def create_embeddings_matrix(word_embeddings, vocab):
    embedding_size = len(next(iter(word_embeddings.values())))
    vocab_size = len(vocab)

    W_emb = np.zeros((vocab_size, embedding_size))

    nb_unk = 0
    for i, t in vocab.id2token.items():
        if i == Vocab.PAD_TOKEN:
            W_emb[i] = np.zeros((embedding_size,))
        else:
            if t in word_embeddings:
                W_emb[i] = word_embeddings[t]
            else:
                W_emb[i] = np.random.uniform(-0.3, 0.3, embedding_size)
                nb_unk += 1

    print(f'Unknown tokens: {nb_unk}')
    print(f'W_emb: {W_emb.shape}')

    return W_emb


def create_word_embeddings(cfg, vocab):
    word_embeddings_filename = None

    if cfg.word_embeddings == WordEmbeddings.GloVe:
        word_embeddings_filename = 'glove.840B.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.MIMIC:
        word_embeddings_filename = 'mimic.fastText.no_clean.300d.pickled'

    word_embeddings_filename = cfg.word_embeddings_dir.joinpath(word_embeddings_filename)
    word_embeddings = load_pickle(word_embeddings_filename)
    print(f'Embeddings: {len(word_embeddings)}')

    W_emb = create_embeddings_matrix(word_embeddings, vocab)

    return W_emb


def create_model(cfg, model_params, **kwargs):
    model_class = None
    model_params.update(kwargs)

    if cfg.model == Models.SimpleModel:
        model_class = models.SimpleModel

    model = model_class(**model_params)
    init_weights(model)
    model = to_device(model)

    print(f'Model: {model.__class__.__name__}')

    return model


def get_dataset(cfg):
    if not cfg.cache_dir.exists():
        cfg.cache_dir.mkdir()

    cache_filename = cfg.cache_dir.joinpath(f'dataset_{int(cfg.lowercase)}_{cfg.max_len}.pkl')
    if not cache_filename.exists():
        mednli_train, mednli_dev, mednli_test = load_mednli(cfg)

        dataset_train = NLIDataset(mednli_train, lowercase=cfg.lowercase, max_len=cfg.max_len)
        dataset_dev = NLIDataset(mednli_dev, vocab=dataset_train.vocab, lowercase=cfg.lowercase, max_len=cfg.max_len)
        dataset_test = NLIDataset(mednli_test, vocab=dataset_train.vocab, lowercase=cfg.lowercase, max_len=cfg.max_len)

        save_pickle((dataset_train, dataset_dev, dataset_test,), cache_filename)
    else:
        dataset_train, dataset_dev, _ = load_pickle(cache_filename)

    print(f'Dataset: {len(dataset_train)} - {len(dataset_dev)},  Vocab: {len(dataset_train.vocab)}')

    return dataset_train, dataset_dev