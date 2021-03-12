import sys
sys.path.append("/home/jack/torchstudy/01Jan/2_refactoring/src")
sys.path.append("/home/jack/torchstudy/02Feb/2_rafactoring/src")
from vocab import Vocabs
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import linecache
import torch
from torch.nn.utils.rnn import pad_sequence

class DeEndataset(Dataset):
    """
    https://github.com/pytorch/text/issues/130
    """
    def __init__(self, data_paths):
        """
        data_paths = list of de path, en data path
        """
        self.de_path = data_paths[0] 
        self.en_path = data_paths[1]

        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.src_vocab = Vocabs(self.de_tokenizer)
        self.dst_vocab = Vocabs(self.en_tokenizer)

        self.src_vocab.update_vocabs_to_file(self.de_path)
        self.dst_vocab.update_vocabs_to_file(self.en_path)

        self.src_vocab.set_most_common_dict(size = 6000)
        self.dst_vocab.set_most_common_dict(size = 6000)

        with open(self.de_path, "r") as f:
            self._total_data = len(f.readlines()) - 1

    def __len__(self):
        return self._total_data 

    def __getitem__(self, idx):     
        raw_de = linecache.getline(self.de_path, idx + 1).replace("\n","")
        raw_en = linecache.getline(self.en_path, idx + 1).replace("\n","")
        order = -1
        de_tensor_ = torch.tensor([self.src_vocab.vocab_dict[token] for token in self.src_vocab.tokenizer(raw_de.lower())[::order]]).long()
        en_tensor_ = torch.tensor([self.dst_vocab.vocab_dict[token] for token in self.dst_vocab.tokenizer(raw_en.lower())]).long()
        return (de_tensor_, en_tensor_)
    
    @classmethod
    def collate_fn(cls, data_batch, pad_idx=0, sos_idx=2, eos_idx=3):
        de_batch, en_batch = [], []
        de_len, en_len = [] , []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([sos_idx]), de_item, torch.tensor([eos_idx])], dim=0))
            en_batch.append(torch.cat([torch.tensor([sos_idx]), en_item, torch.tensor([eos_idx])], dim=0))
            de_len.append(len(de_batch[-1]))

        sorted_v, sort_i = torch.Tensor(de_len).sort(descending = True)
        padded_de_batch = pad_sequence(de_batch, padding_value=pad_idx)
        padded_en_batch = pad_sequence(en_batch, padding_value=pad_idx)

        padded_sorted_de_batch = padded_de_batch.T[sort_i].T
        padded_sorted_en_batch = padded_en_batch.T[sort_i].T

        sorted_de_len = sorted_v.long()
        return (padded_sorted_de_batch, sorted_de_len), padded_sorted_en_batch

class KoEndataset(Dataset):
    """
    https://github.com/pytorch/text/issues/130
    """
    def __init__(self, data_paths):
        """
        data_paths = list of de path, en data path
        """
        self.ko_path = data_paths[0]
        self.en_path = data_paths[1]

        self.ko_tokenizer = "khaiii" #khaiii 등록
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.src_vocab = Vocabs(self.ko_tokenizer)
        self.dst_vocab = Vocabs(self.en_tokenizer)

        self.src_vocab.update_vocabs_to_file(self.ko_path)
        self.dst_vocab.update_vocabs_to_file(self.en_path)

        self.src_vocab.set_most_common_dict(size = 6000)
        self.dst_vocab.set_most_common_dict(size = 6000)

        with open(self.ko_path, "r") as f:
            self._total_data = len(f.readlines()) - 1

    def __len__(self):
        return self._total_data 

    def __getitem__(self, idx):     
        raw_ko = linecache.getline(self.ko_path, idx + 1).replace("\n","")
        raw_en = linecache.getline(self.en_path, idx + 1).replace("\n","")
        order = -1
        ko_tensor_ = torch.tensor([self.src_vocab.vocab_dict[token] for token in self.src_vocab.tokenizer(raw_ko.lower())[::order]]).long() #khaiii 형식에 맞게
        en_tensor_ = torch.tensor([self.dst_vocab.vocab_dict[token] for token in self.dst_vocab.tokenizer(raw_en.lower())]).long()
        return (ko_tensor_, en_tensor_)
    
    @classmethod
    def collate_fn(cls, data_batch, pad_idx=0, sos_idx=2, eos_idx=3):
        ko_batch, en_batch = [], []
        ko_len, en_len = [] , []
        for (de_item, en_item) in data_batch:
            ko_batch.append(torch.cat([torch.tensor([sos_idx]), ko_item, torch.tensor([eos_idx])], dim=0))
            en_batch.append(torch.cat([torch.tensor([sos_idx]), en_item, torch.tensor([eos_idx])], dim=0))
            ko_len.append(len(ko_batch[-1]))

        sorted_v, sort_i = torch.Tensor(ko_len).sort(descending = True)
        padded_ko_batch = pad_sequence(ko_batch, padding_value=pad_idx)
        padded_en_batch = pad_sequence(en_batch, padding_value=pad_idx)

        padded_sorted_ko_batch = padded_ko_batch.T[sort_i].T
        padded_sorted_en_batch = padded_en_batch.T[sort_i].T

        sorted_ko_len = sorted_v.long()
        return (padded_sorted_ko_batch, sorted_ko_len), padded_sorted_en_batch

if __name__ == "__main__":
    train_data_paths = [
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.train.ko",
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.train.en"
    ]

    valid_data_paths = [
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.dev.ko",
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.dev.en"
        ]

    test_data_paths = [
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.test.ko",
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.test.en"
        ]

    BATCH_SIZE = 3
    
    TrainDataset = KoEndataset(train_data_paths)
    TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    ValidDataset = KoEndataset(valid_data_paths)
    ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    TestDataset = KoEndataset(test_data_paths)
    TestDataloader = DataLoader(TestDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    for i in TrainDataloader:
        # print(i[0][0].size())
        # print(i[0][1].size(), i[0][1].tolist())
        print(i[0][0])
        print(i[0][1])
        print(i[1].size())
        break