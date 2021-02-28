from konlpy.tag import Mecab
import numpy as np
from collections import defaultdict


class Vocabs:
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = lambda x: x.split(" ")
        elif tokenizer == "mecab":
            self.tokenizer = Mecab().morphs
        else:
            self.tokenizer = tokenizer

        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self._index = 4
        self.vocab_dict = defaultdict(lambda: self.unk_idx)
        self.vocab_dict["<PAD>"] = self.pad_idx
        self.vocab_dict["<UNK>"] = self.unk_idx
        self.vocab_dict["<SOS>"] = self.sos_idx
        self.vocab_dict["<EOS>"] = self.eos_idx


    def __len__(self):
        return len(self.vocab_dict)

    def update_vocabs_to_file(self, filepath):
        with open(filepath, encoding="utf8") as f:
            for string_ in f:
                for token in self.tokenizer(string_.replace("\n","").lower()):
                    if token in self.vocab_dict:
                        pass
                    else:
                        self.vocab_dict[token] = self._index
                        self._index += 1

    def __len__(self):
        return len(self.vocab_dict)

    def build_vocabs(self, sentence_list):
        for sentence in sentence_list:
            tokens_list = self.tokenizer(sentence)
            for word in tokens_list:
                if word in self.vocab_dict:
                    pass
                else:
                    self.vocab_dict[word] = self._index
                    self._index += 1

    def build_index_dict(self):
        self.index_dict = {v: k for k, v in self.vocab_dict.items()}

    def stoi(self, sentence, option=None, reverse=False):
        if option == "seq2seq":
            if type(sentence) == str:
                if reverse == True:
                    return [self.sos_idx] + [self.vocab_dict[word] for word in self.tokenizer(sentence)][::-1] + [self.eos_idx]
                return [self.sos_idx] + [self.vocab_dict[word] for word in self.tokenizer(sentence)] + [self.eos_idx]
            elif type(sentence) == list:
                return [self.stoi(i, option=option, reverse=reverse) for i in sentence]

        else:
            if type(sentence) == str:
                return [self.vocab_dict[word] for word in self.tokenizer(sentence)]
            elif type(sentence) == list:
                return [self.stoi(i) for i in sentence]

    def itos(self, indices):
        if type(indices[0]) == int:
            return " ".join([self.index_dict[index] for index in indices if self.index_dict[index] != "<PAD>"])
        elif type(indices) == list:
            return [self.itos(i) for i in indices]

    def init_vectors(self, emb_dim):
        self.word_vecs = dict()
        for word in self.vocab_dict:
            self.word_vecs[word] = np.random.uniform(-0.25, 0.25, emb_dim)
