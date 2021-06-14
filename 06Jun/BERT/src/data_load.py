import os
import linecache
import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import random
import torch


def load_tokenizer(tokenizer_path):
    loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, strip_accents=False, lowercase=False)  # Must be False if cased model  # 로드
    return loaded_tokenizer


class KoDataset_single_sentence(Dataset):
    def __init__(self, vocab_txt_path, txt_path):
        self.tokenizer = load_tokenizer(vocab_txt_path)
        self.txt_path = txt_path

        with open(self.txt_path, "r") as f:
            self._total_data = len(f.readlines())

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        raw_ko = linecache.getline(self.txt_path, idx + 1).strip()
        return raw_ko

    def collate_fn(self, batch, max_seq_len):
        batch = self.tokenizer(batch, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")["input_ids"]
        batch = batch.permute(1, 0)
        return batch


class KoDataset_nsp_mlm(Dataset):
    def __init__(self, vocab_txt_path, txt_path, nsp_prob=0.5, mask_prob=0.15):
        self.tokenizer = load_tokenizer(vocab_txt_path)
        self.txt_path = txt_path
        self.nsp_prob = nsp_prob
        self.mask_prob = mask_prob
        self.max_length = 128
        with open(self.txt_path, "r") as f:
            self._total_data = len(f.readlines())

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        now_ko = linecache.getline(self.txt_path, idx + 1).strip()
        if random.random() > self.nsp_prob:
            next_ko = linecache.getline(self.txt_path, idx + 2).strip()
            nsp = torch.Tensor([1]).long()
        else:
            next_ko = linecache.getline(self.txt_path, random.randint(0, self._total_data)).strip()
            nsp = torch.Tensor([0]).long()

        seq = [now_ko, next_ko]
        input_ids = self.tokenizer.encode(seq, max_length=self.max_length, truncation=True, return_tensors="pt").squeeze()
        labels = input_ids.detach().clone()

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != 2) * (input_ids != 3)
        input_ids[mask_arr] = int(self.tokenizer.mask_token_id)

        return input_ids, mask_arr, labels, nsp

    def collate_fn(self, batch):
        ret_dict = dict()
        max_padded_len = max([i.size()[0] for i, _, _, _ in batch])
        max_padded_len = min(max_padded_len, 512)

        batch_inputs = torch.zeros((len(batch), max_padded_len)).long()
        batch_labels = torch.zeros((len(batch), max_padded_len)).long()
        batch_mask_tokens = torch.zeros((len(batch), max_padded_len)).long()
        batch_attention_masks = torch.zeros((len(batch), max_padded_len)).long()
        batch_nsp = torch.zeros(len(batch))

        for idx, (input_ids, mask_arr, labels, nsp) in enumerate(batch):
            for _id, _v in enumerate(input_ids):
                if _v == 4:
                    if 0.1 < random.random() < 0.2:
                        _changed_number = random.choice(range(5, self.tokenizer.vocab_size))
                        input_ids[_id] = _changed_number
                        # print(_v, _changed_number, "마스크 -> 아무 토큰")
                    elif random.random() <= 0.1:
                        # print(_v, labels[_id].item(), "마스크 -> 원래 토큰")
                        input_ids[_id] = labels[_id].item()
                    else:
                        # print(_v, "마스크 -> 마스크")
                        pass

            batch_inputs[idx, : len(input_ids)] = input_ids
            batch_mask_tokens[idx, : len(mask_arr)] = mask_arr
            batch_attention_masks[idx, : len(input_ids)] = 1
            batch_labels[idx, : len(input_ids)] = labels
            batch_nsp[idx] = nsp

        ret_dict["masked_inputs"] = batch_inputs.long()
        ret_dict["attention_masks"] = batch_attention_masks.long()
        ret_dict["labels"] = batch_labels.long()
        ret_dict["nsp_labels"] = batch_nsp.long()
        ret_dict["mask_marking"] = batch_mask_tokens.long()

        return ret_dict


# Label을 포함한 TC 데이터셋
class KoDataset_with_label_ynat(KoDataset_single_sentence):
    def __init__(self, *args, **kwargs):
        KoDataset_single_sentence.__init__(self, *args, **kwargs)
        self.label_dict = {"생활문화": 0, "사회": 1, "IT과학": 2, "스포츠": 3, "세계": 4, "정치": 5, "경제": 6}

    def __getitem__(self, idx):
        label, raw_ko = linecache.getline(self.txt_path, idx + 1).strip().split("\t")
        return [label, raw_ko]

    def collate_fn(self, batch, max_seq_len):
        labels = []
        sents = []
        for label, raw_ko in batch:
            labels.append(self.label_dict[label])
            sents.append(raw_ko)
        labels = torch.Tensor(labels).long()
        bert_batch = self.tokenizer(sents, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")["input_ids"]
        bert_batch = bert_batch.permute(1, 0)
        return bert_batch, labels


if __name__ == "__main__":
    # _single = False
    # _bert = True
    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    txt_path = "/home/jack/torchstudy/06Jun/BERT/data/wpm/sentence_cleaned_namu_train.txt"

    # max_seq_len = 256
    # if _single:
    #     dataset = KoDataset_single_sentence(vocab_txt_path, txt_path)
    #     train_data_loader = DataLoader(dataset, collate_fn= lambda batch : dataset.collate_fn(batch, max_seq_len), batch_size=3)

    dataset = KoDataset_nsp_mlm(vocab_txt_path, txt_path)
    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=10, shuffle=True)

    # else:
    #     dataset = KoDataset_with_label_ynat(vocab_txt_path, txt_path)
    #     train_data_loader = DataLoader(dataset, collate_fn= lambda batch : dataset.collate_fn(batch, max_seq_len), batch_size=3)

    for i in train_data_loader:
        # print(i)
        print("Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in i["labels"][0].tolist()]))
        break
