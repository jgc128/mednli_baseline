import random
import string
from datetime import datetime

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
        nb_classes=len(NLIDataset.LABEL_TO_ID),
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


def load_word_embeddings(cfg):
    word_embeddings_filename = None
    if cfg.word_embeddings == WordEmbeddings.GloVe:
        word_embeddings_filename = 'glove.840B.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.MIMIC:
        word_embeddings_filename = 'mimic.fastText.no_clean.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.BioAsq:
        word_embeddings_filename = 'bio_asq.no_clean.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.WikiEn:
        word_embeddings_filename = 'wiki_en.fastText.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.WikiEnMIMIC:
        word_embeddings_filename = 'wiki_en_mimic.fastText.no_clean.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.GloVeBioAsq:
        word_embeddings_filename = 'glove_bio_asq.no_clean.300d.pickled'
    if cfg.word_embeddings == WordEmbeddings.GloVeBioAsqMIMIC:
        word_embeddings_filename = 'glove_bio_asq_mimic.no_clean.300d.pickled'

    word_embeddings_filename = cfg.word_embeddings_dir.joinpath(word_embeddings_filename)
    word_embeddings = load_pickle(word_embeddings_filename)
    print(f'Embeddings: {len(word_embeddings)}')

    return word_embeddings


def create_word_embeddings(cfg, vocab):
    word_embeddings = load_word_embeddings(cfg)

    W_emb = create_embeddings_matrix(word_embeddings, vocab)

    return W_emb


def create_model(cfg, model_params, **kwargs):
    model_class = None
    model_params = model_params.copy()
    model_params.update(kwargs)

    if cfg.model == Models.Simple:
        model_class = models.SimpleModel
    if cfg.model == Models.InferSent:
        model_class = models.InferSentModel
    if cfg.model == Models.ESIM:
        model_class = models.ESIMModel

    model = model_class(**model_params)
    init_weights(model)
    model = to_device(model)

    print(f'Model: {model.__class__.__name__}')

    return model


def get_dataset(cfg):
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


def randomize_name(name, include_date=False):
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if include_date:
        current_date = datetime.now().strftime('%Y-%m-%d')
        full_name = f'{name}.{current_date}.{random_suffix}'
    else:
        full_name = f'{name}.{random_suffix}'

    return full_name


def create_dirs(cfg):
    target_dirs = [cfg.cache_dir, cfg.models_dir, ]

    for target_dir in target_dirs:
        if not target_dir.exists():
            target_dir.mkdir()
