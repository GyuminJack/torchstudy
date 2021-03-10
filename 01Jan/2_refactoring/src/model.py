import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
import ray


class CNN1d(nn.Module):
    def __init__(self, model_type, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout_rate, pad_idx, **kwargs):
        super().__init__()
        self.model_type = model_type

        if model_type == "multichannel":
            self.embedding_static = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_static.weight.requires_grad = False
            self.embedding_nonstatic = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embs = [self.embedding_static, self.embedding_nonstatic]

        elif model_type == "static":
            self.embedding_static = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_static.weight.requires_grad = False
            self.embs = [self.embedding_static]

        elif model_type == "non-static":
            self.embedding_nonstatic = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embs = [self.embedding_nonstatic]

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    @classmethod
    def make_init_vectors(cls, w2v_path, my_vocabs):
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        lower = -(np.std(w2v_model.vectors, axis=0) ** 2)
        upper = lower * -1
        my_data_vectors = []
        w2v_tokens = w2v_model.vocab.keys()
        for token in my_vocabs:
            if token in w2v_tokens:
                my_data_vectors.append(torch.FloatTensor(w2v_model[token]))
            else:
                my_data_vectors.append(torch.FloatTensor(np.random.uniform(lower, upper)))
        stacked_data = torch.stack(my_data_vectors)
        return stacked_data

    def _init_embedding_vectors(self, w2v_path, my_vocabs):
        if self.model_type == "non-static":
            vectors_in_trainset = []
            for _, v in my_vocabs.word_vecs.items():
                vectors_in_trainset.append(torch.FloatTensor(v))
            [emb.weight.data.copy_(torch.stack(vectors_in_trainset)) for emb in self.embs]
        elif w2v_path != None:
            vectors_in_trainset = self.make_init_vectors(w2v_path, my_vocabs.vocab_dict.keys())
            [emb.weight.data.copy_(vectors_in_trainset) for emb in self.embs]

    def _ray_embedding_vectors(self, ray_w2v_id, my_vocabs):
        w2v_model = ray.get(ray_w2v_id)
        lower = -(np.std(w2v_model.vectors, axis=0) ** 2)
        upper = lower * -1
        my_data_vectors = []
        w2v_tokens = w2v_model.vocab.keys()
        for token in my_vocabs.vocab_dict.keys():
            if token in w2v_tokens:
                my_data_vectors.append(torch.FloatTensor(w2v_model[token]))
            else:
                my_data_vectors.append(torch.FloatTensor(np.random.uniform(lower, upper)))
        stacked_data = torch.stack(my_data_vectors)
        [emb.weight.data.copy_(stacked_data) for emb in self.embs]

    def forward(self, text):
        embs = [emb(text).permute(0, 2, 1) for emb in self.embs]
        if len(embs) == 1:
            conved = [F.relu(conv(embs[0])) for conv in self.convs]
        elif len(embs) == 2:
            conved = [F.relu(conv(embs[0])) + F.relu(conv(embs[1])) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
