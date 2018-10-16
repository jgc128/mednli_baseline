import numpy as np
import torch
import torch.utils.data


def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj, (list, tuple)):
        return [to_device(o, device=device) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = obj.to(device)
    return obj


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def get_trainable_parameters(parameters):
    parameters = list(parameters)
    nb_params_before = sum(p.nelement() for p in parameters)

    parameters = [p for p in parameters if p.requires_grad]
    nb_params_after = sum(p.nelement() for p in parameters)

    print(f'Parameters: {nb_params_before} -> {nb_params_after}')
    return parameters


def create_data_loader(dataset, batch_size, shuffle=True):
    pin_memory = torch.cuda.is_available()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return data_loader


def load_weights(model, filename):
    map_location = None

    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage

    state_dict = torch.load(str(filename), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)

    print(f'Model loaded: {filename}')


def save_weights(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), str(filename))

    print(f'Model saved: {filename}')


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking).long()

    lengths = masks.sum(dim=dim)

    return lengths


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, return_sequence=False):
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequence = return_sequence
        self.bidirectional = bidirectional

        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                                 batch_first=True)

    def zero_state(self, inputs):
        batch_size = inputs.size()[0]

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1 if not self.bidirectional else 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        h0, c0 = (inputs.new_zeros(state_shape), inputs.new_zeros(state_shape),)

        return h0, c0

    def forward(self, inputs, lengths=None):
        h0, c0 = self.zero_state(inputs)

        if lengths is not None:
            # sort by length
            lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
            inputs_sorted = inputs[inputs_sorted_idx]

            # pack sequences
            packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, list(lengths_sorted.data), batch_first=True)

            outputs, (h, c) = self.rnn(packed, (h0, c0))

            # unpack sequences
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            # un-sort
            _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
            outputs = outputs[inputs_unsorted_idx]

            # concat in case of bidirectional, and just remove the first dim in case of unidirectional
            h = torch.cat([x for x in h], dim=-1)
            h = h[inputs_unsorted_idx]
        else:
            outputs, (h, c) = self.rnn(inputs, (h0, c0))

            # concat in case of bidirectional, and just remove the fisrt dim in case of unidirectional
            # h = torch.cat(h, dim=-1)
            h = torch.cat([x for x in h], dim=-1)

        if self.return_sequence:
            return outputs
        else:
            return h
