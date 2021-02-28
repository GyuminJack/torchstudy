import sys
sys.path.append("/home/jack/torchstudy/1week/2_refactoring/src")
sys.path.append("/home/jack/torchstudy/2week/2_rafactoring/src")
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

        sorted_de_batch = sorted(de_batch, key=lambda x :len(x), reverse = True)
        sorted_de_len = torch.Tensor(sorted(de_len, reverse=True)).long()

        padded_sorted_de_batch = pad_sequence(sorted_de_batch, padding_value=pad_idx)
        padded_en_batch = pad_sequence(en_batch, padding_value=pad_idx)
    
        return (padded_sorted_de_batch, sorted_de_len), padded_en_batch

if __name__ == "__main__":
    train_data_paths = [
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/train.de",
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/train.en"
    ]

    valid_data_paths = [
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/val.de",
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/val.en"
        ]

    test_data_paths = [
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/test2016.de",
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/test2016.en"
        ]

    BATCH_SIZE = 3
    
    TrainDataset = DeEndataset(train_data_paths)
    TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    ValidDataset = DeEndataset(valid_data_paths)
    ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    TestDataset = DeEndataset(test_data_paths)
    TestDataloader = DataLoader(TestDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    for dataloader in [TrainDataloader, ValidDataloader, TestDataloader]:
        for item in dataloader:
            print(item)
            break