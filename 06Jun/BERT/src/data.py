import os
import linecache
import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import random
import torch
import numpy as np
import math

def load_tokenizer(tokenizer_path):
    loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, strip_accents=False, lowercase=False)  # Must be False if cased model  # 로드
    return loaded_tokenizer


class KoDataset_single_sentence(Dataset):
    def __init__(self, vocab_txt_path, txt_path, *args, **kwargs):
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
    def __init__(self, vocab_txt_path, txt_path, nsp_prob=0.5, mask_prob=0.3):
        self.tokenizer = load_tokenizer(vocab_txt_path)
        self.txt_path = txt_path
        self.nsp_prob = nsp_prob
        self.mask_prob = mask_prob
        self.max_length = 256
        with open(self.txt_path, "r") as f:
            self._total_data = len(f.readlines())

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        now_ko = linecache.getline(self.txt_path, idx + 1).strip()
        if random.random() > self.nsp_prob:
            next_ko = linecache.getline(self.txt_path, idx + 2).strip()
            nsp = 1
        else:
            rand_idx = random.randint(0, self._total_data)
            # 같은 문장이거나, 다음 문장일 경우 다시 뽑기
            while (rand_idx == idx+1) or (rand_idx == idx+2):
                rand_idx = random.randint(0, self._total_data)
            next_ko = linecache.getline(self.txt_path, rand_idx).strip()
            nsp = 0
            
        seq = [now_ko, next_ko]
        input_ids = self.tokenizer.encode(seq, max_length=self.max_length, truncation=True, return_tensors="pt").squeeze()
        labels = input_ids.detach().clone()

        exclude_cls_sep = ((input_ids != 2) & (input_ids != 3)) * torch.arange(0, len(input_ids)).long()
        mask_target = exclude_cls_sep[exclude_cls_sep != 0]
        pick_mask_index = sorted(np.random.choice(mask_target, math.ceil(len(mask_target) * self.mask_prob), replace=False))
        
        segment_embedding = torch.ones(input_ids.size()).long()
        segment_embedding[(min((input_ids == 3).nonzero(as_tuple=True)[0])+1):] = 2

        mask_arr = torch.zeros(input_ids.size()).long()
        mask_arr[pick_mask_index] = 1

        input_ids[pick_mask_index] = int(self.tokenizer.mask_token_id)
        # print(input_ids)
        mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        
        def _change_masks_index(mask_index_list):
            mask_index_list = list(mask_index_list)
            tot = len(mask_index_list)
            mask_cnt = math.ceil(tot * 0.8)
            
            if tot - mask_cnt == 1:
                random_cnt = 1
            else:
                random_cnt = math.ceil((tot-mask_cnt)*0.5)

            mask_index = list(np.random.choice(mask_index_list, mask_cnt, replace=False))
            random_index = list(np.random.choice(list(set(mask_index_list)-set(mask_index)), random_cnt, replace=False))
            original_index = list(set(mask_index_list)-set(mask_index)-set(random_index))

            return mask_index, random_index, original_index
        
        _, _r, _o = _change_masks_index(mask_indices.tolist())
        
        for _ir in _r:
            input_ids[_ir] = random.choice(range(5, self.tokenizer.vocab_size))

        for _io in _o:
            input_ids[_io] = labels[_io].item()

        return input_ids, segment_embedding, mask_arr, labels, nsp

    def collate_fn(self, batch):
        ret_dict = dict()
        max_padded_len = max([i.size()[0] for i, _, _, _, _ in batch])
        max_padded_len = min(max_padded_len, 512)

        batch_inputs = torch.zeros((len(batch), max_padded_len)).long()
        batch_segments = torch.zeros((len(batch), max_padded_len)).long()
        batch_labels = torch.zeros((len(batch), max_padded_len)).long()
        batch_mask_tokens = torch.zeros((len(batch), max_padded_len)).long()
        batch_attention_masks = torch.zeros((len(batch), max_padded_len)).long()
        batch_nsp = torch.zeros(len(batch))

        for idx, (input_ids, segment_embedding, mask_arr, labels, nsp) in enumerate(batch):

            batch_inputs[idx, : len(input_ids)] = input_ids
            batch_segments[idx, : len(segment_embedding)] = segment_embedding
            batch_mask_tokens[idx, : len(mask_arr)] = mask_arr
            batch_attention_masks[idx, : len(input_ids)] = 1
            batch_labels[idx, : len(input_ids)] = labels
            batch_nsp[idx] = nsp

        ret_dict["masked_inputs"] = batch_inputs.long()
        ret_dict["segment_embedding"] = batch_segments.long()
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
        self.max_seq_len = 256
        
    def __getitem__(self, idx):
        label, raw_ko = linecache.getline(self.txt_path, idx + 1).strip().split("\t")
        return [label, raw_ko]

    def collate_fn(self, batch):
        labels = []
        sents = []
        for label, raw_ko in batch:
            labels.append(self.label_dict[label])
            sents.append(raw_ko)
        labels = torch.Tensor(labels).long()
        inputs = self.tokenizer(sents, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        return inputs, labels


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

    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=1, shuffle=True)

    # else:
    #     dataset = KoDataset_with_label_ynat(vocab_txt_path, txt_path)
    #     train_data_loader = DataLoader(dataset, collate_fn= lambda batch : dataset.collate_fn(batch, max_seq_len), batch_size=3)

    for i in train_data_loader:
        print(i)
        print("Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in i["labels"][0].tolist()]))
        break
