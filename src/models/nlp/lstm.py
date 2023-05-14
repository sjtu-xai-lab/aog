import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, text, text_lengths):
        # text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        if self.lstm.bidirectional:
            # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
            # and apply dropout
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # hidden = [batch size, hid dim * 2]
        else:
            hidden = self.dropout(hidden[-1, :, :])  # hidden = [batch size, hid dim]

        return self.fc(hidden)

    def forward_perturb(self, text, text_lengths, perturbation):
        embedded = self.dropout(self.embedding(text))
        embedded = embedded + perturbation
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        _, (hidden, _) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)

    def get_emb(self, text):
        embedded = self.dropout(self.embedding(text))
        return embedded

    def emb2out(self, embedded, length_tensor=None):
        # embedded = [batch size, sent len, emb dim]
        # assert embedded.shape[0] == 1, f"Expected shape [1, sen_len, emb_dim], but {embedded.shape}"
        if length_tensor is None:
            length_tensor = torch.LongTensor([embedded.shape[1]] * embedded.shape[0])
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length_tensor.cpu(), batch_first=True)
        _, (hidden, _) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # hidden = [batch size, hid dim * 2]
        else:
            hidden = self.dropout(hidden[-1, :, :])  # hidden = [batch size, hid dim]
        return self.fc(hidden)

    def to_eval_mode(self):
        self.eval()
        self.lstm.train()


def lstm2_uni(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
    return LSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=2,
        bidirectional=False,
        dropout=0.5,
        pad_idx=pad_idx
    )