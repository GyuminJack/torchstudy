import sys
try:
    from src.vocab import Vocabs
except:
    from vocab import Vocabs
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
import linecache
import torch
from torch.nn.utils.rnn import pad_sequence

class KoDataset(Dataset):
    def __init__(self, data_path):
        """
        data_paths = list of de path, en data path
        """
        self.ko_path = data_path 
        self.ko_vocab = Vocabs(tokenizer = 'mecab')
        self.ko_vocab.update_vocabs_to_file(self.ko_path)
        self.id2word_dict = self.ko_vocab.get_index_dict()
        self.character_dict = self.ko_vocab.get_character_dict()

        with open(self.ko_path, "r") as f:
            self._total_data = len(f.readlines())
        
    def __len__(self):
        return self._total_data 

    def __getitem__(self, idx):     
        raw_ko = linecache.getline(self.ko_path, idx + 1).strip()
        ko_tensor_ = torch.tensor([self.ko_vocab.vocab_dict[token] for token in self.ko_vocab.tokenizer(raw_ko)]).long()
        return ko_tensor_
    
    def collate_fn(self, data_batch, pad_idx=0, sos_idx=2, eos_idx=3):
        ko_batch = []
        char_batch = []

        token_max_len = 10
        max_seq_len = 0
        for each_item in data_batch:
            # token_max_len = max(token_max_len, max([len(self.id2word_dict[int(i)]) for i in each_item]))
            max_seq_len = max(max_seq_len, len(each_item))

        for each_item in data_batch:
            ko_batch.append(each_item)
            chars = []
            for index in each_item:
                words = []
                word = self.id2word_dict[int(index)]
                for char in word:
                    words.append(self.character_dict[char])
                if len(words) > token_max_len:
                    words = words[:token_max_len]
                else:
                    words = words + [0 for _ in range(token_max_len-len(words))]
                chars.append(words)

            chars = chars + [[0 for _ in range(token_max_len)] for _ in range(max_seq_len - len(chars))]
            char_batch.append(chars)
        padded_ko_index_batch = pad_sequence(ko_batch, padding_value=pad_idx, batch_first=True)
        padded_ko_char_batch = torch.Tensor(char_batch).long()
        return padded_ko_index_batch, padded_ko_char_batch

