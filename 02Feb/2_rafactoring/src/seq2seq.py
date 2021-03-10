import torch

# Encoder pack input : torch.Size([36, 128, 1000]) seq * batch * emb_size
# Decoder LSTM input : torch.Size([1, 128, 1000]) target_id * batch * emb_size
# Decoder hidden_state input : torch.Size([4, 128, 1000]) n_layers * batch * hid_size
# Decoder cell_state input : torch.Size([4, 128, 1000]) n_layers * batch * cell_size

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, pad_idx = 0, dropout=0.5):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src, src_len):
        #src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        # embedded = [batch size,src len, emb dim]
        # # Pack padded batch of sequences for RNN module
        # print(f"Encoder pack input : {embedded.size()}")
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len.tolist(), batch_first=False)
        packed_output, (hidden, cell) = self.rnn(packed_input)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, output_dim, n_layers, pad_idx=0, dropout=0.5):
        super().__init__()
        self.emb = torch.nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = torch.nn.Linear(hid_dim, output_dim)
        self.output_dim = output_dim
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, encoder_hidden_state, encoder_cell_state):
        # 디코더에는 배치 하나가 들어오기 때문에 차원이 하나 축소되서 들어옴 -> 이걸 한차원 늘리려고 하는 작업
        if x.dim() == 2:
            x = x
        else:
            x = x.unsqueeze(0)
        embedded = self.dropout(self.emb(x)) # batch, seq_len, emb_size
        # encoder_hidden_state(input) : [num_layer * num_direction, batch, hidden_size ] 

        # print(f"Decoder LSTM input : {embedded.size()}")
        # print(f"Decoder hidden_state input : {encoder_hidden_state.size()}")
        # print(f"Decoder cell_state input : {encoder_cell_state.size()}")
        output, (hidden, cell) = self.rnn(embedded, (encoder_hidden_state, encoder_cell_state))
        # out : [ sequence_len, batch_size, num_direction * hidden_size ]. (batch_first = False)
        # hidden : [ num_layer * num_direction, batch, hidden_size]
        # cell : [ num_layer * num_direction, batch, hidden_size]
        fc_out = self.fc_out(output.squeeze(0))
        return fc_out, hidden, cell


class seq2seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, src_len, trg):
        
        hidden, cell = self.encoder(src, src_len)
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        
        # first input to the decoder is the <sos> tokens
        # trg : <sos> token1 token2 ... <eos>
        input = trg[0,:] # 처음값 세팅
        
        for t in range(1, trg_len): # 마지막 입력 없음.
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output # output[0] 에는 아무것도 담기지 않음.
            input = trg[t]
        return outputs

    def predict(self, inputs, sos_token=2, eos_idx=3):
        with torch.no_grad():
            hidden, cell = self.encoder.predict(inputs)
        outs = []
        probs = []

        now_token = torch.Tensor([[sos_token]]).long()
        trial = 0
        break_point = 100
        with torch.no_grad():
            while True:
                embedded = seq2seq.decoder.emb(now_token)
                output, (hidden, cell) = seq2seq.decoder.lstm(embedded, (hidden, cell))
                output = seq2seq.decoder.fc_out(output.squeeze(0))
                next_index = int(output.argmax())
                if next_index == eos_idx:
                    break

                outs.append(next_index)
                now_token = torch.Tensor([[next_index]]).long()
                # probs.append(float(torch.nn.Softmax(dim=2)(output).max()))

                trial += 1
                if trial == break_point:
                    break
        return outs, probs
