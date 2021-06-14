import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors

class BaseGRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, output_dim, w2v_path, n_layers = 2, dropout_rate = 0.5, pad_idx = 0, device = 'cuda:1'):
        super(BaseGRUClassifier, self).__init__()
        vocab_size = len(vocab)
        self.embedding_static = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.embs = [self.embedding_static]

        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, bidirectional = True)
        self.fc = nn.Linear(hid_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        if w2v_path is not None:
            self._init_embedding_vectors(w2v_path, vocab)
            self.embedding_static.weight.requires_grad = False
 
    def forward(self, tokens):
        independent_input = tokens[1].to(self.device)
        
        embedding = self.embedding_static(independent_input)
        output, hidden = self.rnn(embedding)
        # output = (seq_len, batch, num_directions * hidden_size)
        last_hiddens = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim = -1)
        fc_out = self.fc(last_hiddens)
        return fc_out

    @classmethod
    def make_init_vectors(cls, w2v_path, vocab_dict):
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path)
        lower = -(np.std(w2v_model.vectors, axis=0) ** 2)
        upper = lower * -1
        my_data_vectors = []
        w2v_tokens = w2v_model.vocab.keys()
        for token in vocab_dict:
            if token in w2v_tokens:
                my_data_vectors.append(torch.FloatTensor(np.copy(w2v_model[token])))
            else:
                my_data_vectors.append(torch.FloatTensor(np.random.uniform(lower, upper)))
        stacked_data = torch.stack(my_data_vectors)
        return stacked_data

    def _init_embedding_vectors(self, w2v_path, my_vocabs):
        try:
            vectors_in_trainset = self.make_init_vectors(w2v_path, my_vocabs.vocab_dict.keys())
        except:
            vectors_in_trainset = self.make_init_vectors(w2v_path, my_vocabs.keys())
        [emb.weight.data.copy_(vectors_in_trainset) for emb in self.embs]

def get_hiddens(model, input_batch, nlayers=2):
    model.eval()
    with torch.no_grad():
        hiddens = []
        embs = []
        steps = input_batch.size()[1]
        for i in range(steps):
            each_input = input_batch[:,i,:].unsqueeze(1)
            x = model.embedding(each_input) # embedding & highway
            x = x.permute(1, 0, 2)
            output, (hidden, c_state) = model.rnn(x)
            embs.append(x)
            hiddens.append(hidden)
        embs = torch.stack(embs)
        embs = embs.permute(2, 0, 1, 3)
        hiddens = torch.stack(hiddens)
        hiddens = hiddens.permute(2, 0, 1, 3)

        fhiddens, bhiddens = hiddens[:,:,:nlayers,:], hiddens[:,:,nlayers:,:]
        s_lstm_hiddens = torch.cat([fhiddens, bhiddens], dim=3)
        s_embs = torch.cat([embs, embs], dim = 3)
        ret = torch.cat([s_embs, s_lstm_hiddens], dim=2)
    return ret

class task_fine_tune(nn.Module):
    def __init__(self, concat_dim, n_layers = 3, projection = 512, device = 'cpu'):
        super(task_fine_tune, self).__init__() 
        self.device = device
        self.task_gamma = nn.Parameter(torch.ones(1, requires_grad=True))
        self.task_tensor = nn.Parameter(torch.ones(n_layers, 1 , requires_grad=True))
        self.task_params = [self.task_gamma, self.task_tensor]
        self.softmax = nn.Softmax(dim=0)
        self.projection = False

        if projection == -1:
            self.projection = False

        elif projection > 0:
            self.projection = True
            self.fc = nn.Linear(concat_dim, projection)

    def forward(self, input):
        input = input.to(self.device)
        task_s = self.softmax(self.task_tensor).to(self.device)
        task_vectors = torch.einsum('lt,bsle->bse', task_s, input)
        task_vectors = torch.einsum('g,bse->bse', self.task_gamma, task_vectors)
        if self.projection:
            task_vectors = self.fc(task_vectors)
        task_vectors = task_vectors.permute(1, 0, 2)
        return task_vectors

class GRUClassifier(nn.Module):
    def __init__(self, elmo, elmo_hidden_output_dim, elmo_projection_dim, elmo_train_vocab, hid_dim, output_dim, w2v_path, emb_dim = 200, use_idp_emb = True, n_layers = 2, dropout_rate = 0.5, pad_idx = 0, device = 'cuda:1'):
        super(GRUClassifier, self).__init__()
        self.elmo = elmo

        vocab_size = len(elmo_train_vocab)
        self.device = device
        
        self.task_emb = task_fine_tune(concat_dim = elmo_hidden_output_dim, projection = elmo_projection_dim, device=device)
        if elmo_projection_dim == -1:
            elmo_projection_dim = elmo_hidden_output_dim

        if use_idp_emb:
            self.use_w2v = True
            self.idp_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
            self.embs = [self.idp_embedding]
            self.rnn = nn.GRU(elmo_projection_dim + emb_dim, hid_dim, num_layers=n_layers, bidirectional = True)
            if w2v_path != 'random':
                self._init_embedding_vectors(w2v_path, elmo_train_vocab)
                self.idp_embedding.weight.requires_grad = False
        else:
            self.use_w2v = False
            self.rnn = nn.GRU(elmo_projection_dim, hid_dim, num_layers=n_layers, bidirectional = True)

        self.fc = nn.Linear(hid_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, tokens):
        elmo_character_input, independent_input = tokens[0].to(self.device), tokens[1].to(self.device)
        elmo_hiddens = get_hiddens(self.elmo, elmo_character_input)
        elmo_embeddings = self.task_emb(elmo_hiddens).to(self.device)

        if self.use_w2v:
            independent_embeddings = self.idp_embedding(independent_input).to(self.device)
            embedding_concat = torch.cat([elmo_embeddings, independent_embeddings], dim = -1)
        else:
            embedding_concat = elmo_embeddings
        output, hidden = self.rnn(embedding_concat)
        # output = (seq_len, batch, num_directions * hidden_size)
        
        last_hiddens = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim = -1)
        fc_out = self.fc(last_hiddens)
        return fc_out
    
    @classmethod
    def make_init_vectors(cls, w2v_path, vocab_dict):
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path)
        lower = -(np.std(w2v_model.vectors, axis=0) ** 2)
        upper = lower * -1
        my_data_vectors = []
        w2v_tokens = w2v_model.vocab.keys()
        for token in vocab_dict:
            if token in w2v_tokens:
                my_data_vectors.append(torch.FloatTensor(np.copy(w2v_model[token])))
            else:
                my_data_vectors.append(torch.FloatTensor(np.random.uniform(lower, upper)))
        stacked_data = torch.stack(my_data_vectors)
        return stacked_data

    def _init_embedding_vectors(self, w2v_path, my_vocabs):
        vectors_in_trainset = self.make_init_vectors(w2v_path, my_vocabs.vocab_dict.keys())
        [emb.weight.data.copy_(vectors_in_trainset) for emb in self.embs]
