import logging

import numpy as np
import scipy.spatial.distance
import distance
import nltk


def words_in_common(sentence1, sentence2):
    if len(sentence1) == 0 or len(sentence2) == 0:
        return 0

    question1_words = set(sentence1)
    question2_words = set(sentence2)

    words_share = len(question1_words & question2_words) / (len(question1_words | question2_words))

    return words_share


def consecutive_words_last_different(sentence1, sentence2):
    if len(sentence1) == 0 or len(sentence2) == 0:
        return 0

    i = 0
    for i, (t1, t2) in enumerate(zip(sentence1, sentence2)):
        if t1 != t2:
            break

    r = i / min(len(sentence1), len(sentence2))

    return r


def entities_in_common(sentence1, sentence2):
    question1_ents = set(sentence1)
    question2_ents = set(sentence2)

    if len(question1_ents) == 0 or len(question2_ents) == 0:
        return 0

    ent_score = len(question1_ents & question2_ents) / (len(question1_ents | question2_ents))

    return ent_score


def has_entity_other_does_not(sentence1, sentence2):
    question1_ents = set(sentence1)
    question2_ents = set(sentence2)

    if len(question1_ents) == 0 and len(question2_ents) == 0:
        return False

    if len(question1_ents - question2_ents) != 0 or len(question2_ents - question1_ents) != 0:
        return True

    return False


def has_common_entity(sentence1, sentence2):
    question1_ents = set(sentence1)
    question2_ents = set(sentence2)

    has_entity = len(question1_ents & question2_ents) != 0

    return has_entity


def word_match_share_anokas(sentence1, sentence2, stops):
    q1words = {}
    q2words = {}
    for word in sentence1:
        if word not in stops:
            q1words[word] = 1

    for word in sentence2:
        if word not in stops:
            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))

    return R


def nb_tokens(sentence1, sentence2, kind='average', uniq=False, stops=None, char=False):
    if stops is not None:
        sentence1 = [w for w in sentence1 if w not in stops]
        sentence2 = [w for w in sentence2 if w not in stops]

    if uniq:
        sentence1 = set(sentence1)
        sentence2 = set(sentence2)

    if char:
        sentence1 = ''.join(sentence1)
        sentence2 = ''.join(sentence2)

    nb_tokens_q1 = len(sentence1)
    nb_tokens_q2 = len(sentence2)

    nb_tokens_res = 0
    if kind == 'average':
        nb_tokens_res = (nb_tokens_q1 + nb_tokens_q2) / 2
    if kind == 'min':
        nb_tokens_res = min(nb_tokens_q1, nb_tokens_q2)
    if kind == 'max':
        nb_tokens_res = max(nb_tokens_q1, nb_tokens_q2)
    if kind == 'diff':
        nb_tokens_res = abs(nb_tokens_q1 - nb_tokens_q2)
    if kind == 'ratio':
        eps = 0.000001
        if nb_tokens_q1 > nb_tokens_q2:
            nb_tokens_res = nb_tokens_q1 / (nb_tokens_q2 + eps)
        else:
            nb_tokens_res = nb_tokens_q2 / (nb_tokens_q1 + eps)

    return nb_tokens_res


def presence_of_no(sentence1, sentence2):
    question1_no = any(t == 'no' for t in sentence1)
    question2_no = any(t == 'no' for t in sentence2)

    return question1_no != question2_no


def vector_similarity(sentence1, sentence2, kind='cosine'):
    if sentence1.sum() == 0 or sentence2.sum() == 0:
        return 0

    if not isinstance(sentence1, np.ndarray):
        sentence1 = sentence1.todense().flatten()
        sentence2 = sentence2.todense().flatten()

    distance_fn = None
    if kind == 'cosine':
        distance_fn = scipy.spatial.distance.cosine
    if kind == 'euclidean':
        distance_fn = scipy.spatial.distance.euclidean
    if kind == 'manhattan':
        distance_fn = scipy.spatial.distance.cityblock
    if kind == 'chebyshev':
        distance_fn = scipy.spatial.distance.chebyshev

    similarity = 1 - distance_fn(sentence1, sentence2)

    if np.isnan(similarity):
        similarity = 0

    return similarity


def pos_diff(sentence1, sentence2, kind='VERB'):
    nb_pos_q1 = len([p for p in sentence1 if p == kind])
    nb_pos_q2 = len([p for p in sentence2 if p == kind])

    nb_pos_diff = abs(nb_pos_q1 - nb_pos_q2)

    return nb_pos_diff


def distance_levenshtein(question1, question2, kind='shortest'):
    method = 0
    if kind == 'shortest':
        method = 1
    if kind == 'longest':
        method = 2

    dist = distance.nlevenshtein(question1, question2, method=method)

    return dist


def distance_sorensen(sentence1, sentence2):
    dist = distance.sorensen(sentence1, sentence2)

    return dist


def distance_jaccard(sentence1, sentence2):
    dist = distance.jaccard(sentence1, sentence2)

    return dist


def common_words(sentence1, sentence2, stops=None):
    words = set(sentence1) & set(sentence2)

    if stops is not None:
        words = [w for w in words if w not in stops]

    return len(words)


def total_uniq_words(sentence1, sentence2, stops=None):
    uniq_words = set(sentence1) | set(sentence2)

    if stops is not None:
        uniq_words = [w for w in uniq_words if w not in stops]

    return len(uniq_words)


def same_start_word(sentence1, sentence2):
    if len(sentence1) == 0 or len(sentence2) == 0:
        return False

    return sentence1[0] == sentence2[0]


def bleu_score(sentence1, sentence2):
    bleu = nltk.translate.bleu([sentence1,], sentence2)

    return bleu
