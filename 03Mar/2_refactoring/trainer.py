from models import *
from torchtext.data.metrics import bleu_score
import torch.optim as optim

class Trainer:
    def __init__(self, train_config):
        INPUT_DIM = train_config['input_dim']
        OUTPUT_DIM = train_config['output_dim']
        ENC_EMB_DIM = train_config['emb_dim']
        DEC_EMB_DIM = train_config['emb_dim']
        ENC_HID_DIM = train_config['hid_dim']
        DEC_HID_DIM = train_config['emb_dim']
        device = train_config['device']
        pad_token = train_config['pad_idx']
        self.TRG = train_config['TRG']
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        MAX_OUT_DIM = 2
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, MAX_OUT_DIM, attn)

        self.model = Seq2Seq(enc, dec, device).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad_token)

    def train(self, iterator, clip):
        self.model.train()
        epoch_loss = 0
        t_bleu = 0
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()
            output = self.model(src, trg)
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            batch_size = output.shape[1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
                
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            output_tokens = torch.argmax(output, dim=1).view(batch_size, -1).tolist()
            for j, item in enumerate(output_tokens):
                output_tokens[j] = [str(k) for k in item]

            trg_tokens = trg.view(batch_size, -1).tolist()
            for j, item in enumerate(trg_tokens):
                trg_tokens[j] = [[str(k) for k in item]]
                
    #         print("output :", output_tokens[0])
    #         print("target :", trg_tokens[0][0])
            t_bleu += bleu_score(output_tokens, trg_tokens)
            loss = self.criterion(output, trg)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator), t_bleu / len(iterator)

    def evaluate(self, iterator, epoch, trg_vocab, conf_string):
        val_file = open(f"/home/jack/torchstudy/03Mar/2_refactoring/valdata/val.text.{conf_string}.{epoch}", "w")
        self.model.eval()
        
        epoch_loss = 0
        
        t_bleu = 0
        with torch.no_grad():

            for i, batch in enumerate(iterator):

                src = batch.src
                trg = batch.trg

                output = self.model(src, trg, 0) #turn off teacher forcing

                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]
                batch_size = output.shape[1]
                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                
                output_tokens = torch.argmax(output, dim=1).view(batch_size, -1).tolist()
                for j, item in enumerate(output_tokens):
                    output_tokens[j] = [str(k) for k in item]

                trg_tokens = trg.view(batch_size, -1).tolist()
                for j, item in enumerate(trg_tokens):
                    val_file.write(" ".join([trg_vocab.itos[k] for k in item]) + "\n")
                    trg_tokens[j] = [[str(k) for k in item]]
                
                t_bleu += bleu_score(output_tokens, trg_tokens)
                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]

                loss = self.criterion(output, trg)

                epoch_loss += loss.item()
            
        return epoch_loss / len(iterator), t_bleu / len(iterator)
    
    def init_weights(self):
        def __init_weights(m):
            for name, param in m.named_parameters():
                if "attention.attn.weight" in name:
                    nn.init.normal_(param.data, mean=0, std=0.001 ** 2)
                elif ("bias" in name) or ("decoder.attention.v.weight" in name):
                    nn.init.constant_(param.data, 0)
        #         elif "decoder.fc_out.weight" in name :
        #             nn.init.constant_(param.data[:,-620:], 0)
                else:
                    nn.init.normal_(param.data, mean=0, std=0.01 ** 2)
        self.model.apply(__init_weights)