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
        self.cnn_embedding = configs['character_embedding']
        self.emb_dim = configs['word_embedding']
        self.character_set_size = configs['character_set_size']
        self.hidden_size = configs['hidden_size']
        self.highway_layer = configs['highway_layer']
        self.output_dim = configs['output_dim']
        self.kernal_output = configs['cnn_kernal_output']
        self.schedule = configs['schedule']
        self.save_name = configs['save_name']
        if configs['optimizer'] is not None:
            self.optimizer = configs['optimizer']
        else:
            self.optimizer = optim.Adam

        self.lr = configs['lr']
       
        ch = CnnHighwayEmbedding(self.cnn_embedding, self.emb_dim, self.character_set_size, self.kernal_output, highway_layer=self.highway_layer)
        elmo = ELMo(ch, self.emb_dim, self.hidden_size, self.output_dim)

        self.model = elmo
        self.model.to(self.device)
        # self.initialize_weights(self.model)

    def train(self, model, iterator, optimizer, scheduler, criterion, clip):
        
        model.train()
        epoch_loss = 0
        step = 0
        for original, char_input in iterator:
            step += 1
            st = time.time()
            optimizer.zero_grad()
            elmo_input = char_input[:,:-1,:].to(self.device)
            original_trg = original[:,1:].T.to(self.device)
            
            fpred, bpred = model(elmo_input)

            # foutput = fpred.contiguous().view(-1, self.output_dim)
            # boutput = bpred.contiguous().view(-1, self.output_dim)
            # trg = original_trg.reshape(-1)

            flatten_target = original_trg.view(-1)
            # fliped_target = torch.flip(original_trg, dims = [1]).reshape(-1)
            # print(fpred[0])

            forward_loss = criterion(fpred.view(-1, self.output_dim), flatten_target)
            backward_loss = criterion(bpred.view(-1, self.output_dim), flatten_target)

            loss = forward_loss + backward_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            step_loss_val = loss.item()
            epoch_loss += step_loss_val

            # print(f"step_loss : {step_loss_val:.3f}(->{forward_loss.item():.2f}/<-{backward_loss.item():.2f}), {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")
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
        CLIP = 5
        model = self.model
        best_valid_loss = float('inf')

        schedule_opt = self.schedule

        if schedule_opt:
            c_optimizer = torch.optim.Adam(model.parameters(), lr = 1,  betas = (0.9, 0.98), eps=10e-9)
            c_scheduler = WarmupConstantSchedule(c_optimizer, d_model = 512, warmup_steps = 2000)
        else:    
            c_optimizer = self.optimizer(model.parameters(), lr = self.lr, betas = (0.9, 0.98), eps=10e-9)
            # c_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            c_scheduler = None

        criterion = nn.CrossEntropyLoss(ignore_index = 0)

        for epoch in range(N_EPOCHS):
            start_time = time.time()
        #     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            try:
                train_loss = self.train(model, train_iterator, c_optimizer, c_scheduler, criterion, CLIP)
                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                # if (valid_loss < best_valid_loss) & (train_loss < 2) & (valid_loss < 6.5) & (epoch > 30):
                if (train_loss < best_valid_loss) & (epoch > 3):
                    best_valid_loss = train_loss
                    torch.save(model,'./model/{}'.format(self.save_name))
                # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}', flush=True)
            except:
                pass
