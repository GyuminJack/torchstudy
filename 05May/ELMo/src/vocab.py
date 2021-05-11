import numpy as np
from collections import defaultdict, Counter
from konlpy.tag import Mecab
# from khaiii import KhaiiiApi
# api = KhaiiiApi()

# def khaiii_tokenizer(input_string):
#     token_list = []
#     for word in api.analyze(input_string):
#         for x in word.morphs:
#             token_list.append(x.lex)
#     return token_list

class Vocabs:
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = lambda x: x.split(" ")
        elif tokenizer == "mecab":
            self.tokenizer = Mecab().morphs
        else:
            self.tokenizer = tokenizer
        self.index_dict = None

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

        self.character_dict = None

    def __len__(self):
        return len(self.vocab_dict)


    def build_character_dict(self, size = 2000):
        char_cnt_dict = defaultdict(lambda:0)

        ret = defaultdict(lambda: self.unk_idx)
        ret["<PAD>"] = self.pad_idx
        ret["<UNK>"] = self.unk_idx
        ret["<SOS>"] = self.sos_idx
        ret["<EOS>"] = self.eos_idx
        _index = 4
        for i in self.vocab_dict:
            if i not in ['<PAD>','<UNK>','<SOS>','<EOS>']:
                for char in i:
                    char_cnt_dict[char] += 1

        for k, v in Counter(char_cnt_dict).most_common(size):
            ret[k] = _index
            _index += 1

        return ret

    def get_character_dict(self, size = 2000):
        if self.character_dict is None:
            self.character_dict = self.build_character_dict(size = size)
        return self.character_dict

    def set_most_common_dict(self, size):
        new_dict = defaultdict(lambda: self.unk_idx)
        new_dict["<PAD>"] = self.pad_idx
        new_dict["<UNK>"] = self.unk_idx
        new_dict["<SOS>"] = self.sos_idx
        new_dict["<EOS>"] = self.eos_idx
        _index = 4
        for k, v in Counter(self.count_dict).most_common(size):
            new_dict[k] = _index
            _index += 1
        self.vocab_dict = new_dict

    def update_vocabs_to_file(self, filepath):
        count_dict = defaultdict(lambda: 1)
        with open(filepath, encoding="utf8") as f:
            for string_ in f:
                for token in self.tokenizer(string_.replace("\n","").lower()):
                    if token in self.vocab_dict:
                        count_dict[token] += 1
                        pass
                    else:
                        count_dict[token] = 1
                        self.vocab_dict[token] = self._index
                        self._index += 1
        self.count_dict = count_dict

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

    def get_index_dict(self):
        if self.index_dict == None:
            self.build_index_dict()
        return self.index_dict