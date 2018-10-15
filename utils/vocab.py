from collections import Counter


class Vocab(object):
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, ]

    def __init__(self):
        self.token2id = {}
        self.id2token = {}

        self.tokens_counts = Counter()

        self._add_special_tokens()

    def _add_special_tokens(self):
        self.add_document([Vocab.PAD_TOKEN, Vocab.UNK_TOKEN])

    def add_document(self, document):
        for token in document:
            if token not in self.token2id:
                new_token_id = len(self.token2id)
                self.token2id[token] = new_token_id
                self.id2token[new_token_id] = token

            self.tokens_counts[token] += 1

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    def prune_vocab(self, max_tokens):
        most_common_tokens = [t for t, c in self.tokens_counts.most_common(max_tokens)]
        self.token2id = {t: i + len(Vocab.SPECIAL_TOKENS) for i, t in enumerate(most_common_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

        self._add_special_tokens()

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return len(self.token2id)

    def __str__(self):
        return f'Vocab: {len(self)} tokens'
