import torch

from utils.torch import get_sequences_lengths, LSTMEncoder


class InterSentenceAttention(torch.nn.Module):
    def forward(self, a, b):
        assert a.size()[2] == b.size()[2]
        assert a.size()[0] == b.size()[0]

        # performs batch mat mul
        attention = torch.matmul(a, b.transpose(1, 2), )

        return attention


class InterSentenceInteraction(torch.nn.Module):
    def __init__(self):
        super(InterSentenceInteraction, self).__init__()

        self._attention = InterSentenceAttention()

    def forward(self, a, b, e=None):
        if e is None:
            e = self._attention(a, b)

        e_b = torch.softmax(e, dim=2)
        e_a = torch.softmax(e, dim=1)

        a_tilda = torch.matmul(e_b, b)
        b_tilda = torch.matmul(e_a.transpose(2, 1), a)

        return a_tilda, b_tilda


class InteractionEnhancement(torch.nn.Module):
    def __init__(self, extended=True):
        super(InteractionEnhancement, self).__init__()

        self.extended = extended

    def forward(self, *args):
        to_concat = []
        to_concat.extend(args)

        if self.extended:
            a0 = args[0]
            for a1 in args[1:]:
                to_concat.append(a0 - a1)
                to_concat.append(a0 * a1)

        m_a = torch.cat(to_concat, dim=-1)

        return m_a


class MaxAvgPool(torch.nn.Module):
    def forward(self, inputs, lengths=None):
        inputs_max, _ = inputs.max(dim=1, keepdim=False)

        if lengths is None:
            inputs_avg = inputs.mean(dim=1, keepdim=False)
        else:
            inputs_avg = inputs.sum(dim=1, keepdim=False) / lengths.float().unsqueeze(1)

        result = torch.cat([inputs_avg, inputs_max], dim=-1)

        return result


class ESIMModel(torch.nn.Module):
    def __init__(self, hidden_size, dropout, vocab_size, embedding_size, trainable_embeddings, nb_classes, W_emb=None):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.input_encoder = LSTMEncoder(embedding_size, hidden_size, bidirectional=True, return_sequence=True)

        self.inter_sentence_interaction = InterSentenceInteraction()
        self.interaction_enhancement = InteractionEnhancement()

        self.interaction_enhancement_mapping = torch.nn.Linear(hidden_size * 4 * 2, hidden_size)

        self.rnn_composition = LSTMEncoder(hidden_size, hidden_size, return_sequence=True, bidirectional=True)

        self.max_avg_pool = MaxAvgPool()

        self.fc_representation = torch.nn.Linear(hidden_size * 4 * 2, hidden_size)
        self.fc_logits = torch.nn.Linear(hidden_size, nb_classes)

    def forward(self, premise, hypothesis):
        premise_len = get_sequences_lengths(premise)
        hypothesis_len = get_sequences_lengths(hypothesis)

        premise_emb = self.embedding(premise)
        hypothesis_emb = self.embedding(hypothesis)

        a_bar = self.input_encoder(premise_emb, premise_len)
        b_bar = self.input_encoder(hypothesis_emb, hypothesis_len)

        # =============================
        # === local inference =========
        a_tilda, b_tilda = self.inter_sentence_interaction(a_bar, b_bar)

        # enhancement
        m_a = self.interaction_enhancement(a_bar, a_tilda)
        m_b = self.interaction_enhancement(b_bar, b_tilda)

        # =============================
        # === inference composition ===
        m_a_f = torch.relu(self.interaction_enhancement_mapping(m_a))
        m_b_f = torch.relu(self.interaction_enhancement_mapping(m_b))

        # composition with an RNN
        v_a = self.rnn_composition(m_a_f, premise_len)
        v_b = self.rnn_composition(m_b_f, hypothesis_len)

        # pooling
        v_a = self.max_avg_pool(v_a, premise_len)
        v_b = self.max_avg_pool(v_b, hypothesis_len)
        v = torch.cat([v_a, v_b], dim=-1)

        # ==============================
        # === logits ===================
        v = torch.tanh(self.fc_representation(v))
        logits = self.fc_logits(v)

        return logits
