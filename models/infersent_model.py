import torch
import torch.nn.functional as F

from utils.torch import get_sequences_lengths, LSTMEncoder


class InferSentModel(torch.nn.Module):
    def __init__(self, hidden_size, dropout, vocab_size, embedding_size, trainable_embeddings, nb_classes,W_emb=None):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(embedding_size, hidden_size, bidirectional=True, return_sequence=True)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 4 * 2, hidden_size),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, nb_classes),
        )

    def forward(self, premise, hypothesis):
        premise_len = get_sequences_lengths(premise)
        hypothesis_len = get_sequences_lengths(hypothesis)

        premise_emb = self.embedding(premise)
        hypothesis_emb = self.embedding(hypothesis)

        premise_enc = self.encoder(premise_emb, premise_len)
        hypothesis_enc = self.encoder(hypothesis_emb, hypothesis_len)

        premise_h = torch.max(premise_enc, dim=1)[0]
        hypothesis_h = torch.max(hypothesis_enc, dim=1)[0]

        h_combined = torch.cat(
            [premise_h, hypothesis_h, torch.abs(premise_h - hypothesis_h), premise_h * hypothesis_h], dim=-1
        )

        logits = self.classifier(h_combined)

        return logits
