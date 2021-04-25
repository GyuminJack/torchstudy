
from src.vocab import Vocabs
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import linecache
import torch
from torch.nn.utils.rnn import pad_sequence

class KoKodataset(Dataset):
    """
    https://github.com/pytorch/text/issues/130
    """
    def __init__(self, data_paths):
        """
        data_paths = list of de path, en data path
        """
        self.ko1_path = data_paths[0]
        self.ko2_path = data_paths[1]

        self.ko1_tokenizer = "khaiii" #khaiii 등록
        # self.ko2_tokenizer = get_tokenizer('spacy', language='ko2_core_web_sm')

        self.src_vocab = Vocabs(self.ko1_tokenizer)
        self.dst_vocab = Vocabs(self.ko1_tokenizer)

        self.src_vocab.update_vocabs_to_file(self.ko1_path)
        self.dst_vocab.update_vocabs_to_file(self.ko2_path)

        self.src_vocab.set_most_common_dict(size = 10000)
        self.dst_vocab.set_most_common_dict(size = 10000)

        with open(self.ko1_path, "r") as f:
            self._total_data = len(f.readlines()) - 1

    def __len__(self):
        return self._total_data 

    def __getitem__(self, idx):     
        raw_ko1 = linecache.getline(self.ko1_path, idx + 1).replace("\n","")
        raw_ko2 = linecache.getline(self.ko2_path, idx + 1).replace("\n","")
        order = 1
        ko1_tensor_ = torch.tensor([self.src_vocab.vocab_dict[token] for token in self.src_vocab.tokenizer(raw_ko1.lower())[::order]]).long() #khaiii 형식에 맞게
        ko2_tensor_ = torch.tensor([self.dst_vocab.vocab_dict[token] for token in self.dst_vocab.tokenizer(raw_ko2.lower())]).long()
        return (ko1_tensor_, ko2_tensor_)
    
    @classmethod
    def collate_fn(cls, data_batch, pad_idx=0, sos_idx=2, eos_idx=3):
        ko1_batch, ko2_batch = [], []
        ko1_len, ko2_len = [] , []
        for (ko1_item, ko2_item) in data_batch:
            ko1_batch.append(torch.cat([torch.tensor([sos_idx]), ko1_item, torch.tensor([eos_idx])], dim=0))
            ko2_batch.append(torch.cat([torch.tensor([sos_idx]), ko2_item, torch.tensor([eos_idx])], dim=0))
            ko1_len.append(len(ko1_batch[-1]))

        sorted_v, sort_i = torch.Tensor(ko1_len).sort(descending = True)
        padded_ko11_batch = pad_sequence(ko1_batch, padding_value=pad_idx)
        padded_ko12_batch = pad_sequence(ko2_batch, padding_value=pad_idx)

        padded_sorted_ko11_batch = padded_ko11_batch.T[sort_i].T
        padded_sorted_ko12_batch = padded_ko12_batch.T[sort_i].T

        sorted_ko11_len = sorted_v.long()
        return (padded_sorted_ko11_batch, sorted_ko11_len), padded_sorted_ko12_batch

    @classmethod
    def batch_collate_fn(cls, data_batch, pad_idx=0, sos_idx=2, eos_idx=3):
        ko1_batch, ko2_batch = [], []
        for (ko1_item, ko2_item) in data_batch:
            ko1_batch.append(torch.cat([torch.tensor([sos_idx]), ko1_item, torch.tensor([eos_idx])], dim=0))
            ko2_batch.append(torch.cat([torch.tensor([sos_idx]), ko2_item, torch.tensor([eos_idx])], dim=0))

        padded_ko11_batch = pad_sequence(ko1_batch, padding_value=pad_idx, batch_first=True)
        padded_ko12_batch = pad_sequence(ko2_batch, padding_value=pad_idx, batch_first=True)

        return padded_ko11_batch, padded_ko12_batch



if __name__ == "__main__":
    train_data_paths = [
        "../data/src.tr",
        "../data/dst.tr"
    ]

    # valid_data_paths = [
    #     "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.dev.ko",
    #     "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.dev.en"
    #     ]

    # test_data_paths = [
    #     "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.test.ko",
    #     "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.test.en"
    #     ]

    BATCH_SIZE = 3
    
    TrainDataset = KoKodataset(train_data_paths)
    TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=TrainDataset.batch_collate_fn)
    
    # ValidDataset = KoEndataset(valid_data_paths)
    # ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    # TestDataset = KoEndataset(test_data_paths)
    # TestDataloader = DataLoader(TestDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    for src, trg in TrainDataloader:
        # print(i[0][0].size())
        # print(i[0][1].size(), i[0][1].tolist())
        print(src)
        print(trg)
        break