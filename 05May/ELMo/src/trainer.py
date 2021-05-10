import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
from src.models import *
from torch.autograd import Variable

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        def lr_lambda(step):
            if step == 0:
                return 0
            lrate = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
            return lrate
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda)

class Trainer:
    def __init__(self, configs):

        self.device = configs['device']
        self.epoch = configs['epochs']
        self.cnn_embedding = configs['cnn_embedding']
        self.emb_dim = configs['emb_dim']
        self.character_dict = configs['character_dict']
        self.hidden_size = configs['hidden_size']
        self.output_dim = configs['output_dim']
        
        ch = CnnHighway(self.cnn_embedding, self.emb_dim, self.character_dict)
        elmo = ELMo(ch, self.emb_dim, self.hidden_size, self.output_dim)

        self.model = elmo
        self.model.to(self.device)
        self.initialize_weights(self.model)

        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, model, iterator, optimizer, scheduler, criterion, clip):
        
        model.train()
        epoch_loss = 0
        for original, char_input in iterator:
            optimizer.zero_grad()

            elmo_input = char_input[:,:-1,:].to(self.device)
            original_trg = original[:,1:].to(self.device)

            fpred, bpred = model(elmo_input)
            forward_loss = criterion(fpred.reshape(-1, self.output_dim), original_trg.reshape(-1))
            backward_loss = criterion(bpred.reshape(-1, self.output_dim), original_trg.reshape(-1))
            loss = forward_loss + backward_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

        
    # def evaluate(self, model, iterator, criterion):
    #     model.eval()
    #     epoch_loss = 0
    #     with torch.no_grad():
        
    #         for i, (src, trg) in enumerate(iterator):
    #             src = src.to(self.device)
    #             trg = trg.to(self.device)
    #             output, _ = model(src, trg[:,:-1])
                
    #             #output = [batch size, trg len - 1, output dim]
    #             #trg = [batch size, trg len]
    #             output_dim = output.shape[-1]
    #             output = output.contiguous().view(-1, output_dim)
    #             trg = trg[:,1:].contiguous().view(-1)
                
    #             #output = [batch size * trg len - 1, output dim]
    #             #trg = [batch size * trg len - 1]
    #             # pred_token = output.argmax(1)
    #             # print(pred_token)
    #             loss = criterion(output, trg)
    #             epoch_loss += loss.item()
            
    #     return epoch_loss / len(iterator)

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def run(self, train_iterator):
        N_EPOCHS = self.epoch
        CLIP = 1
        model = self.model
        best_valid_loss = float('inf')

        schedule_opt = False

        if schedule_opt:
            c_optimizer = torch.optim.Adam(model.parameters(), lr = 1,  betas = (0.9, 0.98), eps=10e-9)
            c_scheduler = WarmupConstantSchedule(c_optimizer, d_model = 256, warmup_steps = 4000)
        else:    
            c_optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005,  betas = (0.9, 0.98), eps=10e-9)
        #     c_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
            c_scheduler = None

        criterion = nn.CrossEntropyLoss(ignore_index = 0)
            
        for epoch in range(N_EPOCHS):
            start_time = time.time()
        #     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            train_loss = self.train(model, train_iterator, c_optimizer, c_scheduler, criterion, CLIP)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # if (valid_loss < best_valid_loss) & (train_loss < 2) & (valid_loss < 6.5) & (epoch > 30):
            # if (train_loss < 2) & (valid_loss < 6.5) & (epoch > 30):
            #     best_valid_loss = valid_loss
            #     torch.save(model.state_dict(),'./model/3_best_model_{}.pt'.format(epoch))
            #     print("save")
            # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
