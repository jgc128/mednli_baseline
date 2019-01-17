from pathlib import Path
import functools

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import Config
from utils.features import bleu_score, words_in_common, word_match_share_anokas, nb_tokens, presence_of_no, \
    vector_similarity, distance_levenshtein, distance_sorensen, common_words, total_uniq_words, same_start_word, \
    distance_jaccard
from utils.helpers import randomize_name, load_word_embeddings
from utils.mednli import load_mednli


def extract_features_from_sample(premise, hypothesis, vectorizer, stops, word_vectors):
    premise_str, hypothesis_str = ' '.join(premise), ' '.join(hypothesis)

    tfidf = vectorizer.transform([premise_str, hypothesis_str]).toarray()
    premise_tfidf = tfidf[0]
    hypothesis_tfidf = tfidf[1]

    premise_cbow = np.sum([word_vectors[t] for t in premise if t in word_vectors], axis=0)
    hypothesis_cbow = np.sum([word_vectors[t] for t in hypothesis if t in word_vectors], axis=0)

    features = {
        'bleu_score': bleu_score(premise, hypothesis),
        'words_in_common': words_in_common(premise, hypothesis),
        # 'consecutive_words': consecutive_words_last_different(premise, hypothesis),
        # 'entities_in_common': entities_in_common(question1.entities, question2.entities),
        # 'has_entity_other_does_not': has_entity_other_does_not(question1.entities, question2.entities),
        # 'has_common_entity': has_common_entity(question1.entities, question2.entities),
        'word_match_share_anokas': word_match_share_anokas(premise, hypothesis, stops),
        'nb_tokens_min': nb_tokens(premise, hypothesis, kind='min'),
        'nb_tokens_max': nb_tokens(premise, hypothesis, kind='max'),
        'nb_tokens_diff': nb_tokens(premise, hypothesis, kind='diff'),
        'presence_of_no': presence_of_no(premise, hypothesis),
        'tfidf_similarity_cosine': vector_similarity(premise_tfidf, hypothesis_tfidf, kind='cosine'),
        'tfidf_similarity_euclidean': vector_similarity(premise_tfidf, hypothesis_tfidf, kind='euclidean'),
        'tfidf_similarity_manhattan': vector_similarity(premise_tfidf, hypothesis_tfidf, kind='manhattan'),
        'tfidf_similarity_chebyshev': vector_similarity(premise_tfidf, hypothesis_tfidf, kind='chebyshev'),
        # 'pos_diff_noun': pos_diff(question1.pos, question2.pos),
        # 'pos_diff_verb': pos_diff(question1.pos, question2.pos),
        # 'pos_diff_adp': pos_diff(question1.pos, question2.pos),
        'dist_levenshtein_short': distance_levenshtein(premise, hypothesis, kind='shortest'),
        'dist_levenshtein_long': distance_levenshtein(premise, hypothesis, kind='longest'),
        'dist_sorensen': distance_sorensen(premise, hypothesis),
        'dist_jaccard': distance_jaccard(premise, hypothesis),
        'common_words': common_words(premise, hypothesis),
        'common_words_stops': common_words(premise, hypothesis, stops=stops),
        'total_uniq_words': total_uniq_words(premise, hypothesis),
        'total_uniq_words_stops': total_uniq_words(premise, hypothesis, stops=stops),
        'nb_tokens_ratio': nb_tokens(premise, hypothesis, kind='ratio'),
        'nb_tokens_diff_uniq': nb_tokens(premise, hypothesis, kind='diff', uniq=True),
        'nb_tokens_ratio_uniq': nb_tokens(premise, hypothesis, kind='ratio', uniq=True),
        'nb_tokens_diff_uniq_stops': nb_tokens(premise, hypothesis, kind='diff', uniq=True, stops=stops),
        'nb_tokens_ratio_uniq_stops': nb_tokens(premise, hypothesis, kind='ratio', uniq=True,
                                                stops=stops),
        'same_start_word': same_start_word(premise, hypothesis),
        'char_diff': nb_tokens(premise, hypothesis, kind='diff', char=True),
        'char_ratio': nb_tokens(premise, hypothesis, kind='ratio', char=True),
        'char_diff_uniq_stops': nb_tokens(premise, hypothesis, kind='diff', uniq=True, stops=stops,
                                          char=True),
        'word_vectors_similarity_cosine': vector_similarity(premise_cbow, hypothesis_cbow, kind='cosine'),
        'word_vectors_similarity_euclidean': vector_similarity(premise_cbow, hypothesis_cbow, kind='euclidean'),
        'word_vectors_similarity_manhattan': vector_similarity(premise_cbow, hypothesis_cbow, kind='manhattan'),
        'word_vectors_similarity_chebyshev': vector_similarity(premise_cbow, hypothesis_cbow, kind='chebyshev'),
    }

    return features


def create_feature_matrix(data, vectorizer, stops, word_vectors, label_encoder):
    features = [
        extract_features_from_sample(premise, hypothesis, vectorizer=vectorizer, stops=stops, word_vectors=word_vectors)
        for premise, hypothesis, label
        in tqdm(data, desc='Extracting features')
    ]
    features = pd.DataFrame(features)

    labels = [label for premise, hypothesis, label in data]
    labels = label_encoder.transform(labels)

    return features, labels


def train_model(model, X, y):
    model.fit(X, y)

    return model


def eval_model(model, X, y):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    return acc


def get_model_name(cfg):
    model_name = f'mednli.{cfg.word_embeddings.name.lower()}.{cfg.lowercase}'

    return model_name


def main(cfg):
    model_name = get_model_name(cfg)
    model_name = randomize_name(model_name)
    print(f'Model name: {model_name}')

    # load dataset
    mednli_train, mednli_dev, _ = load_mednli(cfg)

    # load word embeddings
    word_embeddings = load_word_embeddings(cfg)

    # fit vectorizer on both premises and hypothesis from the training set
    premise_train = [' '.join(premise) for premise, hypothesis, label in mednli_train]
    hypothesis_train = [' '.join(hypothesis) for premise, hypothesis, label in mednli_train]
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), lowercase=cfg.lowercase)
    vectorizer.fit(premise_train + hypothesis_train)
    print(f'Vectorizer: {len(vectorizer.get_feature_names())} features')

    label_encoder = LabelEncoder()
    labels_train = [label for premise, hypothesis, label in mednli_train]
    label_encoder.fit(labels_train)
    print(f'Label encoder: {label_encoder.classes_} classes')

    # use nltk stopwords
    stops = set(stopwords.words('english'))
    print(f'Stop words: {len(stops)}')

    # create data matrices
    X_train, y_train = create_feature_matrix(mednli_train, vectorizer, stops, word_embeddings, label_encoder)
    X_dev, y_dev = create_feature_matrix(mednli_dev, vectorizer, stops, word_embeddings, label_encoder)
    print(f'Train: {X_train.shape}, {y_train.shape}, dev: {X_dev.shape}, {y_dev.shape}')

    models = [
        LogisticRegression(solver='liblinear', multi_class='auto'),
        GradientBoostingClassifier(n_estimators=300),
    ]
    for model in models:
        model = train_model(model, X_train, y_train)
        acc_train = eval_model(model, X_train, y_train)
        acc_dev = eval_model(model, X_dev, y_dev)

        print(f'Model: {model.__class__.__name__}, accuracy train: {acc_train:0.3f}, accuracy dev: {acc_dev:0.3f}')


if __name__ == '__main__':
    data_dir = Path(__file__).parent.joinpath('data/')
    cfg = Config(data_dir=data_dir)

    main(cfg)
