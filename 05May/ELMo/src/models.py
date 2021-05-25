import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmbedding(nn.Module):
    def __init__(self, emb_dim, output_dim, character_set_size, kernal_output, pad_idx = 0):
        super(CNNEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(character_set_size, emb_dim, padding_idx=pad_idx)
        self.kernal_out_dims = kernal_output # [[1, 2], [2, 5], [5, 100]]
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=emb_dim, out_channels=fs[1], kernel_size=fs[0]) for fs in self.kernal_out_dims])
        self.dropout = nn.Dropout(0.5)

        # layer normalization 추가
        self.ln = nn.LayerNorm(sum([j for _, j in self.kernal_out_dims]))

        self.fc = nn.Linear(sum([j for _, j in self.kernal_out_dims]), output_dim)

    def forward(self, char_batch):
        # inputs = [Batch_size, max_seq_len, max_character_len]
        a = self.char_embedding(char_batch)

        a = self.dropout(a)
        # embedding = [batch, max_seq_len, max_character_len, character_emb_dim]
        batch_size, seq_len, max_character_len, character_emb_dim = a.size()
        
        # b = [batch * max_seq_len, max_character_len, character_emb_dim]
        b = a.reshape(-1, max_character_len, character_emb_dim)
        
        # b = [batch * max_seq_len, character_emb_dim, max_charater_len]
        b = b.permute(0, 2, 1)
        
        # conv(b) = [batch * max_seq_len, out_channel, max_seq_len - filter_size + 1]
        b = [conv(b) for conv in self.convs]

        # torch.max(conv) = [batch * max_seq_len, out_channel]
        b = [torch.max(conv, dim = -1)[0] for conv in b]

        # torch.cat = [batch * max_seq_len, sum_of_out_channel]
        b = torch.cat(b, dim = 1)

        # b = self.ln(b)

        # b = self.fc(b)

        # b = [batch, max_seq_len, output_dim]
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

class CnnHighwayEmbedding(nn.Module):
    def __init__(self, cnn_emb_dim, elmo_in_dim, character_set_size, kernal_output, highway_layer = 2, highway_func = torch.relu, pad_idx = 0):
        super(CnnHighwayEmbedding, self).__init__()
        self.cnn_emb = CNNEmbedding(cnn_emb_dim, elmo_in_dim, character_set_size, kernal_output, pad_idx = 0)
        self.ln = nn.LayerNorm([elmo_in_dim])
        self.highway = Highway(elmo_in_dim, highway_layer, highway_func)
    
    def forward(self, input):
        x = self.cnn_emb(input)
        x = self.ln(x)
        x = self.highway(x)
        return x


class ELMo(nn.Module):
    def __init__(self, embedding, emb_dim, enc_hid_dim, output_dim, lstm_layer = 2):
        super(ELMo, self).__init__()
        self.embedding = embedding
        # self.ln = nn.LayerNorm([emb_dim])
        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=lstm_layer, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim, output_dim)

    def forward(self, input):
        # x = [batch, max_seq_len, emb_dim]
        x = self.embedding(input) # embedding & highway
        # x = self.ln(x)
        x = self.dropout(x)

        x = x.permute(1, 0, 2)
        output, (hidden, c_state) = self.rnn(x)

        seq_len, batch = output.size()[0:2]

        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size) at token t
        # forward_hidden = output.view(seq_len, batch, 2, -1)[:,:,0,:]
        # backward_hidden = output.view(seq_len, batch, 2, -1)[:,:,1,:]
        
        output = output.reshape(seq_len, batch, -1, 2)
        forward_hidden, backward_hidden = output[:,:,:,0], output[:,:,:,1]

        forward_prediction = self.fc(forward_hidden)
        backward_prediction = self.fc(backward_hidden)
        
        return forward_prediction, backward_prediction