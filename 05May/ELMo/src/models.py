import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmbedding(nn.Module):
    def __init__(self, emb_dim, output_dim, character_dict, pad_idx = 0):
        super(CNNEmbedding, self).__init__()
        self.character_dict = character_dict
        self.char_embedding = nn.Embedding(len(character_dict), emb_dim, padding_idx=pad_idx)
        self.kernal_out_dims = [[3, 3], [3, 5], [4, 10], [5, 10]]
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=emb_dim, out_channels=fs[1], kernel_size=fs[0]) for fs in self.kernal_out_dims])
        
        conv_outdim = 0
        padded_word_len = 10
        for i, _ in self.kernal_out_dims:
            conv_outdim += padded_word_len-i+1

        self.fc = nn.Linear(conv_outdim, output_dim)

    def forward(self, char_batch):
        # inputs = [Batch_size, seq_len]
        a = self.char_embedding(char_batch)
        batch_size, seq_len, max_word_len, emb_dim = a.size()

        b = a.reshape(-1, max_word_len, emb_dim)
        b = [conv(b.permute(0, 2, 1)) for conv in self.convs]
        b = [torch.max(conv.permute(0, 2, 1), dim = -1)[0] for conv in b]
        b = torch.cat(b, dim = 1)

        b = self.fc(b)
        b = b.reshape(batch_size, seq_len, -1)
        return b


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x

class CnnHighway(nn.Module):
    def __init__(self, cnn_emb_dim, elmo_in_dim, character_dict, highway_layer = 2, highway_func = torch.relu, pad_idx = 0):
        super(CnnHighway, self).__init__()
        self.cnn_emb = CNNEmbedding(cnn_emb_dim, elmo_in_dim, character_dict)
        self.highway = Highway(elmo_in_dim, highway_layer, highway_func)
    
    def forward(self, input):
        x = self.cnn_emb(input)
        x = self.highway(x)
        return x


class ELMo(nn.Module):
    def __init__(self, embedding, emb_dim, enc_hid_dim, output_dim, lstm_layer = 2):
        super(ELMo, self).__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=lstm_layer, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, output_dim)

    def forward(self, input):
        x = self.embedding(input)
        x = x.permute(1, 0, 2)
        output, hidden = self.rnn(x)
        predictions = self.fc(output)
        return predictions