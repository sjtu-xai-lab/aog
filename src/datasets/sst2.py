import torch
import torchtext
import torchtext.legacy
from torchtext.legacy.data import TabularDataset, BucketIterator
import numpy as np
import os
import os.path as osp


def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch
    return pad_to_min_len


class SST2(object):
    def __init__(self, data_root, min_len=None):
        self.data_root = data_root
        self.dataset_name = "SST-2"
        self.min_len = min_len
        self._is_data_loaded = False

    def _load_data(self):
        # initialize fields
        if self.min_len is not None:
            self.TEXT = torchtext.legacy.data.Field(
                tokenize='spacy',
                tokenizer_language='en_core_web_sm',
                include_lengths=True,
                batch_first=True,
                postprocessing=get_pad_to_min_len_fn(self.min_len)
            )
        else:
            self.TEXT = torchtext.legacy.data.Field(
                tokenize='spacy',
                tokenizer_language='en_core_web_sm',
                include_lengths=True,
                batch_first=True
            )
        self.LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)

        # load dataset
        fields = [("text", self.TEXT), ("label", self.LABEL)]
        self.train_set, self.test_set = TabularDataset.splits(
            path=osp.join(self.data_root, self.dataset_name),
            format="tsv", train="train.tsv", test="dev.tsv",
            fields=fields, skip_header=True
        )

        # build fields
        self.TEXT.build_vocab(self.train_set)
        self.LABEL.build_vocab(self.train_set)

        self._is_data_loaded = True

    def get_dataloader(self, batch_size, **kwargs):
        if not self._is_data_loaded:
            self._load_data()
        train_iterator, test_iterator = BucketIterator.splits(
            (self.train_set, self.test_set),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.text),
        )
        return train_iterator, test_iterator

    def get_data(self):
        if not self._is_data_loaded:
            self._load_data()
        return self.train_set, self.test_set

    def get_fields(self):
        if not self._is_data_loaded:
            self._load_data()
        return self.TEXT, self.LABEL

    def __repr__(self):
        if not self._is_data_loaded:
            return f"The SST-2 dataset, raw data not loaded"
        else:
            return f"The SST-2 dataset, #train: {len(self.train_set)} " \
                   f"| #test: {len(self.test_set)} " \
                   f"| vocab_size: {len(self.TEXT.vocab)} " \
                   f"| label_to_id: {self.LABEL.vocab.stoi}"


if __name__ == '__main__':
    sst2 = SST2("/data1/limingjie/data/NLP")
    _TEXT, _LABEL = sst2.get_fields()
    _train_set, _test_set = sst2.get_data()
    _train_loader, _test_loader = sst2.get_dataloader(batch_size=64)
    print(_LABEL.vocab.stoi)
    print("Train set:", len(_train_set))
    print("Test set:", len(_test_set))
    print("Sample sentence:", _train_set[0].text)
    print("Label of the sample sentence:", _LABEL.vocab.stoi[_train_set[0].label])

    for batch in _test_loader:
        print(batch.text[0].shape)