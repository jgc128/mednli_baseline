import torch
import torch.nn.functional as F

from utils.torch import get_sequences_lengths


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_size, dropout, vocab_size, embedding_size, trainable_embeddings, nb_classes, W_emb=None):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
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

        premise_proj = self.projection(premise_emb)
        hypothesis_proj = self.projection(hypothesis_emb)

        premise_h = torch.sum(premise_proj, dim=1) / premise_len.unsqueeze(-1).float()
        hypothesis_h = torch.sum(hypothesis_proj, dim=1) / hypothesis_len.unsqueeze(-1).float()

        h_combined = torch.cat([premise_h, hypothesis_h], dim=-1)

        logits = self.classifier(h_combined)

        return logits
