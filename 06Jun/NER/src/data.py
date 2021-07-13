import linecache
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence

def load_tokenizer(tokenizer_path):
    loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, strip_accents=False, lowercase=False)  # Must be False if cased model  # 로드
    return loaded_tokenizer

class KlueDataset_NER(Dataset):
    def __init__(self, vocab_txt_path, txt_path, *args, **kwargs):
        self.tokenizer = load_tokenizer(vocab_txt_path)
        self.txt_path = txt_path
        self.bio_dict = {
                        '[PAD]' : 0,
                        'B-DT': 1,
                        'B-LC': 2,
                        'B-OG': 3,
                        'B-PS': 4,
                        'B-QT': 5,
                        'B-TI': 6,
                        'I-DT': 7,
                        'I-LC': 8,
                        'I-OG': 9,
                        'I-PS': 10,
                        'I-QT': 11,
                        'I-TI': 12,
                        'O': 13
                        }
        with open(self.txt_path, "r") as f:
            self._total_data = len(f.readlines())

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        raw_ko = linecache.getline(self.txt_path, idx + 1).strip()
        text, bio_string = raw_ko.split("\t")
        tokenized_text = self.tokenizer(text, return_tensors="pt")['input_ids'][0]
        bio_tensor = torch.Tensor([self.bio_dict[i] for i in bio_string.split(",")])
        return tokenized_text, bio_tensor

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)
        return x, y

if __name__ == "__main__":
    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    file_path = "/home/jack/torchstudy/06Jun/NER/data/namu_2021060809/klue_ner_20210712.train"
    dataset = KlueDataset_NER(vocab_txt_path, file_path)
    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=2, shuffle=True)
    for i, j in train_data_loader:
        print(i)
        print(j)
        break

