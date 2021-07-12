from model import *
from data import *
import logging
import time

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return 1
            lrate = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
            return lrate
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda)

    
class Trainer:
    def __init__(self, configs):
        input_size = configs["input_dim"]
        self.hidden_dim = configs["d_model"]
        output_size = configs["output_dim"]
        n_layers = configs["n_layers"]
        n_heads = configs["n_heads"]
        pf_mid_dim = configs["pf_dim"]
        dropout_rate = configs["dropout"]
        max_length = configs["max_length"]
        
        self.device = configs["device"]
        self.scheduler = configs['scheduler']
        self.epoch = configs["epoch"]
        self.warmup_steps = configs["warmup_steps"]
        self.model = Encoder(input_size, self.hidden_dim, output_size, n_layers, n_heads, pf_mid_dim, dropout_rate)
        
        self.pretrained_model = configs['pretrained_model']
        if self.pretrained_model is not None:
            self.model = torch.load(self.pretrained_model).module
        else:
            self.initialize_weights(self.model)
        
        self.model.to(self.device)
        
        
        self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.set_optimizer(optimizer = configs['optimizer'], lr = configs['lr'], scheduler=self.scheduler)
        self.set_loss()

    def initialize_weights(self, m):
        for name, param in m.named_parameters():
            if ("fc" in name) or ('embedding' in name):
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
            elif "layer_norm" in name:
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.constant_(param.data, 1.0)
            
    def set_loss(self):
        self.nsp_loss = torch.nn.CrossEntropyLoss()
        self.mlm_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def set_optimizer(self, optimizer, lr, scheduler):
        if scheduler is not None:
            self.optimizer = optimizer(self.model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.01)
            self.scheduler = scheduler(self.optimizer, d_model = self.hidden_dim, warmup_steps = self.warmup_steps)
        else:
            self.scheduler = None
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=0.01)

    def train(self, iterator):
        self.model.train()
        nsp_train_acc = 0
        mlm_train_acc = 0
        train_loss = 0
        for m_batch, batch_dict in enumerate(iterator):
            self.optimizer.zero_grad()
            
            input_tokens = batch_dict["masked_inputs"].to(self.device)
            segments = batch_dict["segment_embedding"].to(self.device)
            attention_masks = (input_tokens != 0).unsqueeze(1).unsqueeze(2).to(self.device)
            true_inputs = batch_dict["labels"].to(self.device)
            nsp_labels = batch_dict["nsp_labels"].to(self.device)
            
            mask_marking = batch_dict["mask_marking"].to(self.device)
            indices = (mask_marking.reshape(-1) == 1).nonzero().reshape(-1).to(self.device)
            nsp_output, mlm_output = self.model(input_tokens, attention_masks, segments)

            selected_mlm_true = torch.index_select(true_inputs.reshape(-1), 0, indices)
            selected_mlm_output = torch.index_select(mlm_output.reshape(-1, mlm_output.size(-1)), 0, indices)

            mlm_pred_label = torch.argmax(selected_mlm_output, dim = -1)
            
            mlm_loss = self.mlm_loss(selected_mlm_output, selected_mlm_true)

            # total_mlm_true = true_inputs.masked_fill(mask_marking!=1, 0)[:, 1:]
            # total_mlm_output = mlm_output
            # mlm_loss = self.mlm_loss(total_mlm_output.transpose(1,2), total_mlm_true)

            # print("MASK Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in mlm_input[:20].tolist()]))
            # print("TRUE Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in mlm_true[:20].tolist()]))
            # print(f"{set([dataset.tokenizer.convert_ids_to_tokens(s) for s in mlm_true.tolist()])}")
            # print("PRED Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in _t[:20].tolist()]))
            # print(f"{set([dataset.tokenizer.convert_ids_to_tokens(s) for s in _t.tolist()])}")
            
            nsp_loss = self.nsp_loss(nsp_output, nsp_labels)

            total_loss = nsp_loss + mlm_loss
            train_loss += total_loss.item()
            
            correct = (torch.argmax(nsp_output, dim=-1) == nsp_labels).float().sum()
            
            nsp_train_acc += correct/len(nsp_labels)
            mlm_acc = (mlm_pred_label == selected_mlm_true).float().sum()/len(selected_mlm_true)
            mlm_train_acc += mlm_acc

            total_loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # print(torch.argmax(nsp_output,1).sum(), (torch.argmax(nsp_output,1)==nsp_labels).sum())
            print(f"{m_batch}/{len(iterator)}, {nsp_loss.item():.3f}, {mlm_loss.item():.3f}, {total_loss.item():.3f}, {correct/len(nsp_labels)*100:.2f}, {mlm_acc*100:.2f}", end="\r")
            # logging.debug(f"# of Total Mask token : {len(indices)}")
            # logging.debug(f"Total Output Size (B, Seq, Out_vocab) : {output.size()}")
            # logging.debug(f"Selected MLM output (# of Mask, Out_vocab) : {mlm_output.size()}")
            # logging.debug(f"NSP output (B, 2) : {nsp_output.size()}")

        return train_loss/len(iterator), nsp_train_acc/len(iterator)*100, mlm_train_acc/len(iterator)*100
    
    @torch.no_grad()
    def valid(self, iterator):
        nsp_valid_acc = 0
        mlm_valid_acc = 0
        valid_loss = 0
        self.model.eval()
        for batch_dict in iterator:
            input_tokens = batch_dict["masked_inputs"].to(self.device)
            segments = batch_dict["segment_embedding"].to(self.device)
            attention_masks = (input_tokens != 0).unsqueeze(1).unsqueeze(2).to(self.device)
            true_inputs = batch_dict["labels"].to(self.device)
            nsp_labels = batch_dict["nsp_labels"].to(self.device)
            mask_marking = batch_dict["mask_marking"].to(self.device)

            indices = (mask_marking[:, 1:].reshape(-1) == 1).nonzero().reshape(-1).to(self.device)           
            mlm_true = torch.index_select(true_inputs[:, 1:].reshape(-1), 0, indices)

            nsp_output, mlm_output = self.model(input_tokens, attention_masks, segments)
            # mlm_labels = torch.index_select(input_tokens.reshape(-1), 0, indices)
            selected_mlm_true = torch.index_select(true_inputs.reshape(-1), 0, indices)
            selected_mlm_output = torch.index_select(mlm_output.reshape(-1, mlm_output.size(-1)), 0, indices)

            mlm_pred_label = torch.argmax(selected_mlm_output, dim = -1)
            mlm_loss = self.mlm_loss(selected_mlm_output, selected_mlm_true)

            # flatten_output = output.reshape(-1, output.size(-1))
            nsp_loss = self.nsp_loss(nsp_output, nsp_labels)
            nsp_valid_label = torch.argmax(nsp_output, dim=1)
            correct = (nsp_valid_label == nsp_labels).float().sum()

            nsp_valid_acc += correct/len(nsp_labels)
            mlm_valid_acc += (mlm_pred_label == selected_mlm_true).float().sum()/len(selected_mlm_true)
            total_loss = nsp_loss + mlm_loss
            valid_loss += total_loss.item()

        return valid_loss/len(iterator), nsp_valid_acc/len(iterator)*100, mlm_valid_acc/len(iterator)*100

    def run(self, train_iter, valid_iter, test_iter):
        best_valid_loss = float('inf')
        for i in range(self.epoch):
            st = time.time()
            train_loss, train_nsp_acc, train_mlm_acc = self.train(train_iter)
            valid_loss, nsp_valid_acc, valid_mlm_acc = self.valid(valid_iter)
            print(f"Time : {time.time()-st:.2f}s, TrainLoss : {train_loss:.5f}, Train_NSPAcc: {train_nsp_acc:.2f}%, Train_MLMAcc: {train_mlm_acc:.2f}%, ValidLoss : {valid_loss:.3f}, ValidNspAcc : {nsp_valid_acc:.2f}%, Valid_MLMAcc: {valid_mlm_acc:.2f}%", flush=True)
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model,'../models/best_petition_namu_{}.pt'.format(i))
       

if __name__ == "__main__":
    import sys

    try:
        mode = sys.argv[1]
        if mode == "debug":
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    except:
        pass

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    txt_path = "/home/jack/torchstudy/06Jun/BERT/data/corpus/petition_namu.train.ko"
    valid_path = "/home/jack/torchstudy/06Jun/BERT/data/corpus/petition_valid.txt"

    dataset = KoDataset_nsp_mlm(vocab_txt_path, txt_path, mask_prob=0.15)
    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=64, shuffle=True)

    valid_dataset = KoDataset_nsp_mlm(vocab_txt_path, valid_path, mask_prob=0.15)
    valid_data_loader = DataLoader(valid_dataset, collate_fn=lambda batch: valid_dataset.collate_fn(batch), batch_size=32)

    base_config = {
        "input_dim": dataset.tokenizer.vocab_size,
        "d_model": 768,
        "output_dim": dataset.tokenizer.vocab_size,
        "n_layers" : 12,
        "n_heads" : 12,
        "pf_dim" : 1024,
        "max_length" : 256,
        "dropout" : 0.1
    }

    train_config = {
        "pretrained_model" : "/home/jack/torchstudy/06Jun/BERT/models/best_petition_namu_1.pt",
        "epoch" : 500,
        "optimizer" : torch.optim.AdamW,
        "scheduler" : WarmupConstantSchedule,
        'warmup_steps' : 60000,
        "lr" :  0.00005, #0.0001 -> 0.0005
        "device" : "cuda:0"
    }

    base_config.update(train_config)

    trainer = Trainer(base_config)
    trainer.run(train_data_loader, valid_data_loader, None)