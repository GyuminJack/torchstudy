import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, n_layers, n_heads, pf_dim, dropout, max_length=256, pos_opt='dynamic'):
        super().__init__()
        assert hid_dim % n_heads == 0, "n_head mis-matched"
        self.tok_embedding = nn.Embedding(input_dim, hid_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, hid_dim)
        
        self.pos_opt = pos_opt
        if pos_opt == 'static':
            self.pos_embedding = PositionalEmbedding(hid_dim, max_len = max_length)
        else:
            self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim)

        self.nsp = nn.Linear(hid_dim, 2)
        self.mlm = nn.Linear(hid_dim, output_dim)


    def forward(self, src, src_mask, segment):

        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        if self.pos_opt != 'static':
            pos = self.pos_embedding(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device))
        else:
            src = self.pos_embedding(src)
        
        segment = self.segment_embedding(segment)
        src = self.tok_embedding(src) + pos + segment
        src = self.dropout(src)
        
        for layer in self.layers:
            src = layer(src, src_mask)

        nsp = self.nsp(src[:, 0, :])
        mlm = self.mlm(src)

        # print(mlm.shape)
        return nsp, mlm
    
    def encode(self, src, src_mask, segment):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = self.pos_embedding(torch.arange(1, src_len+1).unsqueeze(0).repeat(batch_size, 1).to(src.device))
        segment = self.segment_embedding(segment)

        src = (self.tok_embedding(src) * self.scale) + pos + segment
        # src = self.dropout(self.pos_embedding((self.tok_embedding(src) * self.scale)))
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        _src = self.self_attn_layer_norm(src)
        _src, _ = self.self_attention(_src, _src, _src, src_mask)

        src = src + self.dropout(_src)
        # src = [batch size, src len, hid dim]

        _src = self.ff_layer_norm(src)
        src = src + self.positionwise_feedforward(_src)

        src = self.dropout(src)
        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        # Query len = Key len = value_len

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # torch.einsum('bhqd,bhdk->bhqk', Q, K.permute(0, 1, 3, 2))
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.gelu = GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(self.gelu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
