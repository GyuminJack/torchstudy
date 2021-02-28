import torch


class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, num_embeddings, lstm_layers, pad_idx=0):
        super().__init__()
        self.emb = torch.nn.Embedding(embedding_dim=emb_dim, num_embeddings=num_embeddings, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(emb_dim, hid_dim, lstm_layers)

    def forward(self, x, len_x):
        embedded = self.emb(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_x.to('cpu'))
        output, (hidden, cell) = self.lstm(packed_embedded)
        # out : [ sequence_len, batch_size, num_direction * hidden_size ]. (batch_first = False)
        # hidden : [ num_layer * num_direction, batch, hidden_size]
        # cell : [ num_layer * num_direction, batch, hidden_size]

        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, num_embeddings, output_dim, lstm_layers, pad_idx=0):
        super().__init__()
        self.emb = torch.nn.Embedding(embedding_dim=emb_dim, num_embeddings=num_embeddings, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(emb_dim, hid_dim, lstm_layers)
        self.fc_out = torch.nn.Linear(hid_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x, encoder_hidden_state, encoder_cell_state):
        embedded = self.emb(x)
        # encoder_hidden_state(input) : [snum_layer * num_direction, batch, hidden_size ] 
        # 

        output, (hidden, cell) = self.lstm(embedded.unsqueeze(0), (encoder_hidden_state, encoder_cell_state))
        # out : [ sequence_len, batch_size, num_direction * hidden_size ]. (batch_first = False)
        # hidden : [ num_layer * num_direction, batch, hidden_size]
        # cell : [ num_layer * num_direction, batch, hidden_size]
        fc_out = self.fc_out(output)
        return fc_out, hidden, cell


class seq2seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.init_weights()
        
    def forward(self, src, src_len, trg):
        
        hidden, cell = self.encoder(src, src_len)

        dec_input = trg[0, :]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(dec_input, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if not, use predicted token
            dec_input = trg[t, :]

        return outputs

    def init_weights(self):
        models = [self.encoder, self.decoder]
        for m in models:
            for name, param in m.named_parameters():
                torch.nn.init.uniform_(param.data, -0.08, 0.08)

    def predict(self, inputs, sos_token=2, eos_idx=3):
        hidden, cell = self.encoder(inputs)
        last_token = 0
        outs = []
        probs = []

        now_token = torch.Tensor([sos_token]).to(torch.int64)
        trial = 0
        break_point = 100
        with torch.no_grad():
            while True:
                output, hidden, cell = self.decoder(now_token, hidden, cell)
                next_index = int(output.argmax())
                if next_index == eos_idx:
                    break

                outs.append(next_index)
                now_token = torch.Tensor([next_index]).to(torch.int64)
                probs.append(float(torch.nn.Softmax(dim=2)(output).max()))

                trial += 1
                if trial == break_point:
                    break
        return outs, probs
