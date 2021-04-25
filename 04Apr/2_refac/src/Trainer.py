import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
from src.model import *
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

# https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index = 1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        ignore_indices = pred != self.ignore_index
        pred.masked_fill(ignore_indices == 0, -1e10)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.masked_fill(ignore_indices == 0, self.smoothing / (self.cls - 1 - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Trainer:
    def __init__(self, configs):

        INPUT_DIM = configs['input_dim']
        OUTPUT_DIM = configs['output_dim']
        SRC_PAD_IDX = configs['src_pad_idx']
        TRG_PAD_IDX = configs['trg_pad_idx']
        self.device = configs['device']

        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = Encoder(INPUT_DIM, 
                    HID_DIM, 
                    ENC_LAYERS, 
                    ENC_HEADS, 
                    ENC_PF_DIM, 
                    ENC_DROPOUT, 
                    self.device)

        dec = Decoder(OUTPUT_DIM, 
                    HID_DIM, 
                    DEC_LAYERS, 
                    DEC_HEADS, 
                    DEC_PF_DIM, 
                    DEC_DROPOUT, 
                    self.device)


        self.model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, self.device).to(self.device)
        self.model.apply(self.initialize_weights)

    def train(self, model, iterator, optimizer, scheduler, criterion, clip):
        
        model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(iterator):
            src = src.to(self.device)
            trg = trg.to(self.device)
            optimizer.zero_grad()            
            output, _ = model(src, trg[:,:-1])
                    
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
                
            output_dim = output.shape[-1]
                
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)
    
    def evaluate(self, model, iterator, criterion):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
        
            for i, (src, trg) in enumerate(iterator):
                src = src.to(self.device)
                trg = trg.to(self.device)
                output, _ = model(src, trg[:,:-1])
                
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                
                #output = [batch size * trg len - 1, output dim]
                #trg = [batch size * trg len - 1]
                # pred_token = output.argmax(1)
                # print(pred_token)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def run(self, train_iterator, valid_iterator):
        N_EPOCHS = 100
        CLIP = 1
        model = self.model
        best_valid_loss = float('inf')

        TRAINING_LR = []

        schedule_opt = False
        label_smooth_opt = False

        if schedule_opt:
            c_optimizer = torch.optim.Adam(model.parameters(), lr = 1,  betas = (0.9, 0.98), eps=10e-9)
            c_scheduler = WarmupConstantSchedule(c_optimizer, d_model = 256, warmup_steps = 1000)
        else:    
            c_optimizer = torch.optim.Adam(model.parameters(), lr = 0.001,  betas = (0.9, 0.98), eps=10e-9)
        #     c_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
            c_scheduler = None

        if label_smooth_opt:
            criterion = LabelSmoothingLoss(len(TRG.vocab), smoothing=0.05)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index = 0)
            
        for epoch in range(N_EPOCHS):
            start_time = time.time()
        #     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            train_loss = self.train(model, train_iterator, c_optimizer, c_scheduler, criterion, CLIP)
            valid_loss = self.evaluate(model, valid_iterator, criterion)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            elif (epoch > 8) & (valid_loss > best_valid_loss) & (best_valid_loss < 1):
                break
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
