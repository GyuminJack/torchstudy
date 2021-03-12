import sys
sys.path.append("/home/jack/torchstudy/01Jan/2_refactoring/src")
sys.path.append("/home/jack/torchstudy/02Feb/2_rafactoring/src")
from seq2seq import Encoder, Decoder, seq2seq
import torch
import torch.optim as optim
import utils 
import time
import math

class s2sTrainer:
    def __init__(self, encoder_conf, decoder_conf, device):

        self.optimizer = None
        self.loss_fn = None
        self.device = device
        self.encoder = Encoder(**encoder_conf)
        self.decoder = Decoder(**decoder_conf)
        self.seq2seq = seq2seq(self.encoder, self.decoder).to(device)
                
        for name, param in self.seq2seq.named_parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)

    def set_optimizer(self, optimizer = None):
        if optimizer is None:
            self.optimizer = optim.Adam(self.seq2seq.parameters())
        else:
            self.optimizer = optimizer(self.seq2seq.parameters())
    
    def set_loss(self, loss_fn = None, ignore_index = 0):
        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index = ignore_index).to(self.device)
        else:
            self.loss_fn = loss_fn().to(self.device)

    def _train(self, train_iterator):
        self.seq2seq.train()
        clip = 2
        epoch_loss = 0
        for i, (src, trg) in enumerate(train_iterator):
            src, src_len = src[0], src[1]

            src = src.to(self.device)
            trg = trg.to(self.device)

            self.optimizer.zero_grad()
            output = self.seq2seq(src, src_len, trg).to(self.device)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = self.loss_fn(output, trg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.seq2seq.parameters(), clip)
            
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_iterator)
    
    def _evaluate(self, valid_iterator):
        self.seq2seq.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (src, trg) in enumerate(valid_iterator):
                src, src_len = src[0], src[1]
                src = src.to(self.device)
                trg = trg.to(self.device)
                output = self.seq2seq(src, src_len, trg).to(self.device)
                
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)

                loss = self.loss_fn(output, trg)
                
                epoch_loss += loss.item()
        return epoch_loss / len(valid_iterator)
    
    def run(self, epochs, train_iterator, valid_iterator):
        if self.optimizer == None:
            self.set_optimizer()
        
        if self.loss_fn == None:
            self.set_loss()
        
        best_valid_loss = float('inf')
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self._train(train_iterator)
            valid_loss = self._evaluate(valid_iterator)
            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.seq2seq.state_dict(), f'seq2seq-model-ko-en.pt')
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == "__main__":
    encoder_config = {
        "emb_dim" : 1000,
        "hid_dim" : 1000,
        "n_layers" : 4,
        "num_embeddings" : 100,
        "pad_idx" : 0 
    }

    decoder_config = {
        "input_dim" : 1000,
        "hid_dim" : 1000,
        "n_layers" : 4,
        "num_embeddings" : 100,
        "pad_idx" : 0 ,
        "output_dim" : 10
    }

    mock = s2sTrainer(encoder_config, decoder_config, device = "cpu")
