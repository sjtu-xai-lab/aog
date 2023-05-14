import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, text, text_lengths_dummy):
        # text = [batch size, sent len]
        embedded = self.embedding(text)  # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def forward_perturb(self, text, text_lengths_dummy, perturbation):
        embedded = self.embedding(text)
        embedded = embedded + perturbation
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

    def get_emb(self, text):
        embedded = self.embedding(text)
        return embedded

    def emb2out(self, embedded, length_tensor=None):
        # embedded = [batch size, sent len, emb dim]
        # assert embedded.shape[0] == 1, f"Expected shape [1, sen_len, emb_dim], but {embedded.shape}"
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

    def to_eval_mode(self):
        self.eval()


def cnn(vocab_size, embedding_dim, n_filters, output_dim, pad_idx):
    return CNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_filters=n_filters,
        filter_sizes=[3, 4, 5],
        output_dim=output_dim,
        dropout=0.5,
        pad_idx=pad_idx
    )