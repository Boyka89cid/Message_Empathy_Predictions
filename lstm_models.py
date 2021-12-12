# Importing Torch Libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


# LSTM Model for Fixed Input
class LSTM_fix_input(torch.nn.Module):
    """
    Initialization of LSTM Model for Fixed Input
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super().__init__()
        self.embedding_vecs = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Embedding Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM Layer
        self.linear = nn.Linear(hidden_dim, output_size)  # Linear Layer
        self.dropout = nn.Dropout(0.2)  # Dropout

    # Forward Method to use while Training and validation
    def forward(self, x, l):
        x = self.embedding_vecs(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


# LSTM Model for Variable Input
class LSTM_var_input(torch.nn.Module):
    """
    Initialization of LSTM Model in Padded Sequence Layer for variable input
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)  # Dropout
        self.embedding_vecs = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Embedding Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM Layer
        self.linear = nn.Linear(hidden_dim, output_size)  # Linear Layer

    # Forward Method to use while Training and validation
    def forward(self, x, s):
        x = self.embedding_vecs(x)
        x = self.dropout(x)
        x_pack = fix_pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


# LSTM Model with Glove Embeddings for Fixed Input
class LSTM_glove_vecs_input(torch.nn.Module):
    """
    Initializing LSTM Model with glove weights in Embedding Layer
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights, output_size):
        super().__init__()
        self.embedding_vecs = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Embedding Layer
        self.embedding_vecs.weight.data.copy_(torch.from_numpy(glove_weights))  # Glove Vecs inserted in Embedding Layer
        self.embedding_vecs.weight.requires_grad = False  # freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM Layer
        self.linear = nn.Linear(hidden_dim, output_size)  # Linear Layer
        self.dropout = nn.Dropout(0.2)  # Dropout

    # Forward Method to use while Training and Testing
    def forward(self, x, l):
        x = self.embedding_vecs(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


# Method for fixed Padding Sequence
def fix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()  # Converted from cuda to cpu
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices, None)
