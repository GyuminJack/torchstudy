import random
import math
import numpy as np
from src.vocab import Vocabs
import torch
import re
import os

import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, np_data):
        self.x_data = np_data[:, 0]
        self.y_data = np_data[:, 1].reshape(-1, 1).astype(int)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.IntTensor(self.x_data[idx]).to(torch.int64)
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


class DataUtil:
    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def train_test_split(self, data, train_ratio=0.9):
        _len = len(data)
        random.shuffle(data)
        train_data = np.array(data[: math.ceil(_len * train_ratio)])
        test_data = np.array(data[math.ceil(_len * train_ratio) :])
        return train_data, test_data

    # pad in collate_fn
    def collate_pad(self, batch):
        batch_x = [t[0] for t in batch]
        batch_y = [t[1] for t in batch]
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1, batch_first=True).to(torch.int64)
        batch_y = torch.FloatTensor(batch_y)
        return batch_x, batch_y


class MR(DataUtil):
    def __init__(self):
        DATA_PATHS = [
            "/home/jack/torchstudy/1week/0_datas/rt-polarity.neg",
            "/home/jack/torchstudy/1week/0_datas/rt-polarity.pos",
        ]
        self.NEGATIVE_DATAS = self.read_data(DATA_PATHS[0], 1)
        self.POSITVIE_DATAS = self.read_data(DATA_PATHS[1], 0)
        self.TOTAL_DATA = self.NEGATIVE_DATAS + self.POSITVIE_DATAS
        self.text_vocabs = None

    def read_data(self, path, label):
        ret = []
        with open(os.path.abspath(path), "r", encoding="ISO-8859-1") as f:
            for line in f.readlines():
                ret.append([self.clean_str(line.replace("\n", "")), label])
        return ret

    def split(self, split_ratio):
        self.train_data, self.test_data = self.train_test_split(self.TOTAL_DATA, split_ratio)

    def make_train_datasets(self, custompad=-1):
        self.text_vocabs = Vocabs()
        self.text_vocabs.build_vocabs(self.train_data[:, 0])
        train_x_values = self.text_vocabs.stoi(self.train_data[:, 0].tolist())
        if custompad != -1:
            for index, each_data in enumerate(train_x_values):
                if len(each_data) >= custompad:
                    train_x_values[index] = each_data[:custompad]
                elif len(each_data) < custompad:
                    train_x_values[index] = each_data + [self.text_vocabs.vocab_dict["<PAD>"]] * (custompad - len(each_data))
        train_y_values = self.train_data[:, 1]
        train_dataset = CustomDataset(np.array([*zip(train_x_values, train_y_values)]))
        return self.text_vocabs, train_dataset

    def make_test_datasets(self):
        if self.text_vocabs is None:
            raise "Build Training Data First"
        test_x_values = self.text_vocabs.stoi(self.test_data[:, 0].tolist())
        test_y_values = self.test_data[:, 1]
        test_dataset = CustomDataset(np.array([*zip(test_x_values, test_y_values)]))
        return test_dataset


class NSMCDataset(MR):
    def __init__(self, data_type):
        if data_type == "Train":
            train_path = "/home/jack/torchstudy/1week/0_datas/ratings_train.txt"
            self.train_data = self.read_documents(train_path)
        self.text_vocabs = None
        # 중복제거

    def read_documents(self, filename):
        with open(filename, encoding="utf-8") as f:
            documents = [line.split("\t") for line in f.read().splitlines()]
            documents = documents[1:]

        _data = np.array([(self.text_cleaning(line[1]), line[2]) for line in documents if self.text_cleaning(line[1])])
        return _data

    def text_cleaning(self, doc):
        # 한국어를 제외한 글자를 제거하는 함수.
        doc = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)
        return doc


if __name__ == "__main__":
    data_obj = NSMCDataset("Train")
    train_vocabs, train_dataset = data_obj.make_train_datasets(custompad=-1)
    print(data_obj.collate_pad)
