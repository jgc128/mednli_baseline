from pathlib import Path
import enum

import dataclasses


@enum.unique
class WordEmbeddings(enum.Enum):
    GloVe = 'glove'
    MIMIC = 'mimic'
    GloVeBioAsqMIMIC = 'glove_bio_asq_mimic'


@enum.unique
class Models(enum.Enum):
    Simple = 'simple'
    InferSent = 'infersent'


@dataclasses.dataclass
class Config:
    data_dir: Path

    model: Models = Models.InferSent
    word_embeddings: WordEmbeddings = WordEmbeddings.MIMIC

    lowercase: bool = True
    max_len: int = 50
    hidden_size: int = 128
    dropout: float = 0.4
    trainable_embeddings: bool = False

    weight_decay: float = 0.00001
    learning_rate: float = 1e-3
    max_grad_norm: float = 5.0
    batch_size: int = 64
    nb_epochs: int = 30

    @property
    def cache_dir(self) -> Path:
        return self.data_dir.joinpath('cache/')

    @property
    def word_embeddings_dir(self) -> Path:
        return self.data_dir.joinpath('word_embeddings/')
