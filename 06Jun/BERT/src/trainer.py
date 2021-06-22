from model import *
from data import *
import logging

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

        self.model = Encoder(input_size, self.hidden_dim, output_size, n_layers, n_heads, pf_mid_dim, dropout_rate)
        self.model.to(self.device)
        
        self.epoch = configs["epoch"]
        self.set_optimizer(optimizer = configs['optimizer'], lr = configs['lr'], scheduler=self.scheduler)
        self.set_loss()

    def set_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def set_optimizer(self, optimizer, lr, scheduler):
        if scheduler is not None:
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=0.01)
            self.scheduler = scheduler(self.optimizer, d_model = self.hidden_dim, warmup_steps = 1000)

        else:
            self.scheduler = None
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=0.01)

    def train(self, iterator):
        self.model.train()
        ret_loss = 0
        for batch_dict in iterator:
            self.optimizer.zero_grad()

            input_tokens = batch_dict["masked_inputs"].to(self.device)
            segments = batch_dict["segment_embedding"].to(self.device)
            attention_masks = batch_dict["attention_masks"].unsqueeze(1).unsqueeze(2).to(self.device)
            true_inputs = batch_dict["labels"].to(self.device)
            nsp_labels = batch_dict["nsp_labels"].to(self.device)
            mask_marking = batch_dict["mask_marking"][:, 1:].to(self.device)

            indices = (mask_marking.reshape(-1) == 1).nonzero().reshape(-1).to(self.device)

            mlm_true = torch.index_select(true_inputs.reshape(-1), 0, indices)
            mlm_labels = torch.index_select(input_tokens.reshape(-1), 0, indices)

            nsp_output, output = self.model(input_tokens, attention_masks, segments)
            # mlm_output = torch.index_select(output.reshape(-1, output.size(-1)), 0, indices)
            # mlm_loss = self.loss(mlm_output, mlm_labels)
            
            # flatten_output = output.reshape(-1, output.size(-1))
            nsp_loss = self.loss(nsp_output, nsp_labels)
            mlm_loss = 0
            total_loss = nsp_loss + mlm_loss

            total_loss.backward()
            ret_loss += total_loss.item()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

#            logging.debug(f"# of Total Mask token : {len(indices)}")
#            logging.debug(f"Total Output Size (B, Seq, Out_vocab) : {output.size()}")
#            logging.debug(f"Selected MLM output (# of Mask, Out_vocab) : {mlm_output.size()}")
#            logging.debug(f"NSP output (B, 2) : {nsp_output.size()}")

        epoch_loss = ret_loss / len(iterator)
        return epoch_loss
    
    @torch.no_grad()
    def valid(self, iterator):
        nsp_valid_acc = 0
        nsp_valid_loss = 0
        self.model.eval()
        for batch_dict in iterator:
            input_tokens = batch_dict["masked_inputs"].to(self.device)
            segments = batch_dict["segment_embedding"].to(self.device)
            attention_masks = batch_dict["attention_masks"].unsqueeze(1).unsqueeze(2).to(self.device)
            true_inputs = batch_dict["labels"].to(self.device)
            nsp_labels = batch_dict["nsp_labels"].to(self.device)
            mask_marking = batch_dict["mask_marking"][:, 1:].to(self.device)

            indices = (mask_marking.reshape(-1) == 1).nonzero().reshape(-1).to(self.device)

            mlm_true = torch.index_select(true_inputs.reshape(-1), 0, indices)
            mlm_labels = torch.index_select(input_tokens.reshape(-1), 0, indices)

            nsp_output, output = self.model(input_tokens, attention_masks, segments)
            # mlm_output = torch.index_select(output.reshape(-1, output.size(-1)), 0, indices)
            # mlm_loss = self.loss(mlm_output, mlm_labels)
            
            # flatten_output = output.reshape(-1, output.size(-1))
            nsp_loss = self.loss(nsp_output, nsp_labels)
            nsp_valid_label = torch.argmax(nsp_output, dim=1)
            correct = (nsp_valid_label == nsp_labels).float().sum()
            nsp_valid_acc += correct/len(nsp_labels)

            mlm_loss = 0
            total_loss = nsp_loss + mlm_loss
            nsp_valid_loss += nsp_loss.item()
        return nsp_valid_loss/len(iterator), nsp_valid_acc/len(iterator)*100

    def run(self, train_iter, valid_iter, test_iter):
        for i in range(self.epoch):
            train_loss = self.train(train_iter)

            nsp_valld_loss, nsp_valid_acc = self.valid(valid_iter)
            print(f"TrainLoss : {train_loss:.5f}, ValidLoss : {nsp_valld_loss:.3f}, ValidAcc : {nsp_valid_acc:.2f}%", flush=True)
        torch.save(self.model,'../models/tmp_saved.pt')
       

if __name__ == "__main__":
    import sys

    try:
        mode = sys.argv[1]
        if mode == "debug":
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    except:
        pass

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    txt_path = "/home/jack/torchstudy/06Jun/BERT/data/wpm/train_bert_corpus.txt"
    valid_path = "/home/jack/torchstudy/06Jun/BERT/data/wpm/valid_bert_corpus.txt"

    dataset = KoDataset_nsp_mlm(vocab_txt_path, txt_path)
    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=128)

    valid_dataset = KoDataset_nsp_mlm(vocab_txt_path, valid_path)
    valid_data_loader = DataLoader(valid_dataset, collate_fn=lambda batch: valid_dataset.collate_fn(batch), batch_size=128)

    base_config = {
        "input_dim": dataset.tokenizer.vocab_size,
        "d_model": 768,
        "output_dim": dataset.tokenizer.vocab_size,
        "n_layers" : 12,
        "n_heads" : 12,
        "pf_dim" : 2048,
        "max_length" : 256,
        "dropout" : 0.1
    }

    train_config = {
        "epoch" : 500,
        "optimizer" : torch.optim.Adam,
        "scheduler" : WarmupConstantSchedule,
        "lr" : 1e-4,
        "device" : "cuda:0"
    }

    base_config.update(train_config)

    trainer = Trainer(base_config)
    trainer.run(train_data_loader, valid_data_loader, None)
