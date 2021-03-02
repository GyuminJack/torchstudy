import torch


class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, num_embeddings, lstm_layers, pad_idx=0):
        super().__init__()
        self.emb = torch.nn.Embedding(embedding_dim=emb_dim, num_embeddings=num_embeddings, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(emb_dim, hid_dim, lstm_layers)

    def forward(self, x):
        embedded = self.emb(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

    def packed_forward(self, x, len_x):
        embedded = self.emb(x) # batch, seq_len, emb_size
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_x.to('cpu'))
        output, (hidden, cell) = self.lstm(packed_embedded)

        # out : [ sequence_len, batch_size, num_direction * hidden_size ]. (batch_first = False)
        # hidden : [ num_layer * num_direction, batch, hidden_size]
        # cell : [ num_layer * num_direction, batch, hidden_size]
        return hidden, cell
    
    def predict(self, x):
        embedded = self.emb(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, num_embeddings, output_dim, lstm_layers, pad_idx=0):
        super().__init__()
        self.emb = torch.nn.Embedding(embedding_dim=emb_dim, num_embeddings=num_embeddings, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(emb_dim, hid_dim, lstm_layers)
        self.fc_out = torch.nn.Linear(hid_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x, encoder_hidden_state, encoder_cell_state):
        # 디코더에는 배치 하나가 들어오기 때문에 차원이 하나 축소되서 들어옴 -> 이걸 한차원 늘리려고 하는 작업
        x = x.unsqueeze(0)
        embedded = self.emb(x) # batch, seq_len, emb_size
        # encoder_hidden_state(input) : [num_layer * num_direction, batch, hidden_size ] 

        output, (hidden, cell) = self.lstm(embedded, (encoder_hidden_state, encoder_cell_state))
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
        
    def forward(self, src, trg):
        
        hidden, cell = self.encoder(src)
        
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
