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

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    
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

        self.model = Encoder(input_size, self.hidden_dim, output_size, n_layers, n_heads, pf_mid_dim, dropout_rate)
        self.model.to(self.device)
        self.initialize_weights(self.model)

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
            self.scheduler = scheduler(self.optimizer, d_model = self.hidden_dim, warmup_steps = 20)
        else:
            self.scheduler = None
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=0.01)

    def train(self, iterator):
        self.model.train()
        nsp_train_acc = 0
        mlm_train_acc = 0
        train_loss = 0
        for batch_dict in iterator:
            self.optimizer.zero_grad()
            
            input_tokens = batch_dict["masked_inputs"].to(self.device)
            segments = batch_dict["segment_embedding"].to(self.device)
            attention_masks = (input_tokens != 0).unsqueeze(1).unsqueeze(2).to(self.device)
            true_inputs = batch_dict["labels"].to(self.device)
            nsp_labels = batch_dict["nsp_labels"].to(self.device)
            
            mask_marking = batch_dict["mask_marking"].to(self.device)
            
            indices = (mask_marking.reshape(-1) == 1).nonzero().reshape(-1).to(self.device)
            
            nsp_output, mlm_output = self.model(input_tokens, attention_masks, segments)
            
            # _ot = torch.argmax(mlm_output, dim = -1)
            
            # # 마스킹 된 입력 토큰들
            # _a = torch.index_select(input_tokens.reshape(-1), 0, indices)


            # mlm_pred_label == 


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
            print(f"{nsp_loss.item():.3f}, {mlm_loss.item():.3f}, {total_loss.item():.3f}, {correct/len(nsp_labels)*100:.2f}, {mlm_acc*100:.2f}", end="\r")
#            logging.debug(f"# of Total Mask token : {len(indices)}")
#            logging.debug(f"Total Output Size (B, Seq, Out_vocab) : {output.size()}")
#            logging.debug(f"Selected MLM output (# of Mask, Out_vocab) : {mlm_output.size()}")
#            logging.debug(f"NSP output (B, 2) : {nsp_output.size()}")

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
        for i in range(self.epoch):
            train_loss, train_nsp_acc, train_mlm_acc = self.train(train_iter)
            valld_loss, nsp_valid_acc, valid_mlm_acc = self.valid(valid_iter)
            print(f"TrainLoss : {train_loss:.5f}, Train_NSPAcc: {train_nsp_acc:.2f}%, Train_MLMAcc: {train_mlm_acc:.2f}%, ValidLoss : {valld_loss:.3f}, ValidNspAcc : {nsp_valid_acc:.2f}%, Valid_MLMAcc: {valid_mlm_acc:.2f}%", flush=True)
        torch.save(self.model,'../models/tmp_saved.pt')
       

if __name__ == "__main__":
    import sys

    try:
        mode = sys.argv[1]
        if mode == "debug":
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    except:
        pass

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/peti_namu_2021062409"
    txt_path = "/home/jack/torchstudy/06Jun/BERT/data/corpus/head_petition.ko"
    valid_path = "/home/jack/torchstudy/06Jun/BERT/data/corpus/petition_valid.txt"

    dataset = KoDataset_nsp_mlm(vocab_txt_path, txt_path)
    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=32, shuffle=True)

    # for batch_dict in train_data_loader:


    #     input_tokens = batch_dict["masked_inputs"]
    #     segments = batch_dict["segment_embedding"]
    #     attention_masks = batch_dict["attention_masks"].unsqueeze(1).unsqueeze(2)

    #     true_inputs = batch_dict["labels"]
    #     nsp_labels = batch_dict["nsp_labels"]

    #     mask_marking = batch_dict["mask_marking"][:, 1:]
        
    #     indices = (mask_marking.reshape(-1) == 1).nonzero()

    #     print("Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in true_inputs[0].tolist()]))
    #     print("Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in input_tokens[0].tolist()]))
    #     print(mask_marking[0])
    #     indices = (mask_marking[0] == 1).nonzero().reshape(-1)
    #     mlm_true = torch.index_select(true_inputs[0,1:].reshape(-1), 0, indices)
    #     print(indices)
    #     print("Tokens (str) : {}".format([dataset.tokenizer.convert_ids_to_tokens(s) for s in mlm_true.tolist()]))
        
    #     break

    valid_dataset = KoDataset_nsp_mlm(vocab_txt_path, valid_path)
    valid_data_loader = DataLoader(valid_dataset, collate_fn=lambda batch: valid_dataset.collate_fn(batch), batch_size=32)

    base_config = {
        "input_dim": dataset.tokenizer.vocab_size,
        "d_model": 768,
        "output_dim": dataset.tokenizer.vocab_size,
        "n_layers" : 8,
        "n_heads" : 12,
        "pf_dim" : 1024,
        "max_length" : 256,
        "dropout" : 0.1
    }

    train_config = {
        "epoch" : 500,
        "optimizer" : torch.optim.AdamW,
        "scheduler" : None,
        "lr" :  0.0001,
        "device" : "cuda:0"
    }

    base_config.update(train_config)

    trainer = Trainer(base_config)
    trainer.run(train_data_loader, valid_data_loader, None)


    # problem
    # # 조사에 너무 붙음..??
    # BEST (mask_ratio(0.5) embedding initialize, head_petition, shuffle=True)
    # TrainLoss : 9.63055, TrainAcc: 48.05%, ValidLoss : 7.836, ValidAcc : 48.51%
    # TrainLoss : 7.79182, TrainAcc: 48.63%, ValidLoss : 7.898, ValidAcc : 50.11%
    # TrainLoss : 7.66414, TrainAcc: 49.90%, ValidLoss : 7.851, ValidAcc : 49.34%
    # TrainLoss : 7.62827, TrainAcc: 51.37%, ValidLoss : 7.871, ValidAcc : 49.74%
    # TrainLoss : 7.62339, TrainAcc: 51.86%, ValidLoss : 7.814, ValidAcc : 48.60%
    # TrainLoss : 7.61793, TrainAcc: 51.76%, ValidLoss : 7.787, ValidAcc : 49.88%
    # TrainLoss : 7.56990, TrainAcc: 48.05%, ValidLoss : 7.968, ValidAcc : 49.28%
    # TrainLoss : 7.55640, TrainAcc: 50.88%, ValidLoss : 7.933, ValidAcc : 49.28%
    # TrainLoss : 7.54108, TrainAcc: 49.71%, ValidLoss : 7.866, ValidAcc : 51.54%
    # TrainLoss : 7.49810, TrainAcc: 48.54%, ValidLoss : 7.821, ValidAcc : 50.30%
    # TrainLoss : 7.48667, TrainAcc: 49.51%, ValidLoss : 7.902, ValidAcc : 49.88%
    # TrainLoss : 7.48806, TrainAcc: 51.56%, ValidLoss : 7.886, ValidAcc : 50.51%
    # TrainLoss : 7.43641, TrainAcc: 51.27%, ValidLoss : 7.918, ValidAcc : 50.41%
    # TrainLoss : 7.44063, TrainAcc: 50.68%, ValidLoss : 7.898, ValidAcc : 51.11%
    # TrainLoss : 7.38499, TrainAcc: 51.27%, ValidLoss : 8.003, ValidAcc : 46.94%
    # TrainLoss : 7.38908, TrainAcc: 47.85%, ValidLoss : 8.007, ValidAcc : 49.24%
    # TrainLoss : 7.33987, TrainAcc: 49.02%, ValidLoss : 7.914, ValidAcc : 48.51%
    # TrainLoss : 7.31919, TrainAcc: 49.51%, ValidLoss : 7.906, ValidAcc : 48.70%
    # TrainLoss : 7.28053, TrainAcc: 52.34%, ValidLoss : 7.948, ValidAcc : 51.33%
    # TrainLoss : 7.26443, TrainAcc: 51.37%, ValidLoss : 8.016, ValidAcc : 50.09%
    # TrainLoss : 7.29277, TrainAcc: 49.22%, ValidLoss : 8.028, ValidAcc : 48.17%
    # TrainLoss : 7.22266, TrainAcc: 50.68%, ValidLoss : 8.177, ValidAcc : 49.84%
    # TrainLoss : 7.17231, TrainAcc: 50.20%, ValidLoss : 8.209, ValidAcc : 49.83%
    # TrainLoss : 7.12245, TrainAcc: 53.91%, ValidLoss : 8.256, ValidAcc : 48.31%
    # TrainLoss : 7.06626, TrainAcc: 51.46%, ValidLoss : 8.342, ValidAcc : 49.34%
    # TrainLoss : 7.01981, TrainAcc: 55.66%, ValidLoss : 8.352, ValidAcc : 52.21%
    # TrainLoss : 6.98784, TrainAcc: 52.34%, ValidLoss : 8.530, ValidAcc : 50.08%
    # TrainLoss : 6.94425, TrainAcc: 52.05%, ValidLoss : 8.517, ValidAcc : 50.69%
    # TrainLoss : 6.88357, TrainAcc: 53.71%, ValidLoss : 8.530, ValidAcc : 50.95%
    # TrainLoss : 6.84394, TrainAcc: 56.74%, ValidLoss : 8.687, ValidAcc : 49.91%
    # TrainLoss : 6.81496, TrainAcc: 55.47%, ValidLoss : 8.826, ValidAcc : 51.80%
    # TrainLoss : 6.75421, TrainAcc: 59.38%, ValidLoss : 8.812, ValidAcc : 49.10%
    # TrainLoss : 6.73734, TrainAcc: 54.88%, ValidLoss : 8.886, ValidAcc : 50.81%
    # TrainLoss : 6.64968, TrainAcc: 58.11%, ValidLoss : 8.866, ValidAcc : 52.08%
    # TrainLoss : 6.61972, TrainAcc: 61.62%, ValidLoss : 9.199, ValidAcc : 49.88%
    # TrainLoss : 6.57997, TrainAcc: 62.60%, ValidLoss : 9.115, ValidAcc : 54.51%
    # TrainLoss : 6.51358, TrainAcc: 63.57%, ValidLoss : 9.515, ValidAcc : 46.16%
    # TrainLoss : 6.48297, TrainAcc: 60.45%, ValidLoss : 9.387, ValidAcc : 51.54%
    # TrainLoss : 6.42692, TrainAcc: 65.43%, ValidLoss : 10.068, ValidAcc : 48.86%
    # TrainLoss : 6.41672, TrainAcc: 65.53%, ValidLoss : 9.806, ValidAcc : 48.56%
    # TrainLoss : 6.34693, TrainAcc: 65.04%, ValidLoss : 9.499, ValidAcc : 52.31%
    # TrainLoss : 6.29390, TrainAcc: 68.55%, ValidLoss : 9.699, ValidAcc : 52.95%
    # TrainLoss : 6.22416, TrainAcc: 68.55%, ValidLoss : 9.826, ValidAcc : 52.62%
    # TrainLoss : 6.18007, TrainAcc: 70.12%, ValidLoss : 9.944, ValidAcc : 49.34%
    # TrainLoss : 6.18619, TrainAcc: 70.12%, ValidLoss : 9.855, ValidAcc : 51.09%
    # TrainLoss : 6.08814, TrainAcc: 74.61%, ValidLoss : 10.336, ValidAcc : 49.64%
    # TrainLoss : 6.06132, TrainAcc: 71.88%, ValidLoss : 10.004, ValidAcc : 56.46%
    # TrainLoss : 5.99415, TrainAcc: 73.05%, ValidLoss : 10.063, ValidAcc : 53.20%
    # TrainLoss : 5.94536, TrainAcc: 75.10%, ValidLoss : 10.467, ValidAcc : 51.68%
    # TrainLoss : 5.92530, TrainAcc: 76.27%, ValidLoss : 10.526, ValidAcc : 52.42%
    # TrainLoss : 5.90533, TrainAcc: 74.80%, ValidLoss : 10.723, ValidAcc : 52.52%
    # TrainLoss : 5.78590, TrainAcc: 79.59%, ValidLoss : 10.816, ValidAcc : 51.25%
    # TrainLoss : 5.77129, TrainAcc: 80.47%, ValidLoss : 10.720, ValidAcc : 50.07%
    # TrainLoss : 5.75172, TrainAcc: 79.10%, ValidLoss : 10.947, ValidAcc : 49.19%
    # TrainLoss : 5.71025, TrainAcc: 79.39%, ValidLoss : 10.659, ValidAcc : 51.39%
    # TrainLoss : 5.63954, TrainAcc: 82.13%, ValidLoss : 11.286, ValidAcc : 50.37%
    # TrainLoss : 5.59743, TrainAcc: 81.35%, ValidLoss : 10.792, ValidAcc : 50.94%
    # TrainLoss : 5.57288, TrainAcc: 83.40%, ValidLoss : 11.068, ValidAcc : 52.47%
    # TrainLoss : 5.54864, TrainAcc: 81.84%, ValidLoss : 11.301, ValidAcc : 52.32%
    # TrainLoss : 5.51992, TrainAcc: 82.81%, ValidLoss : 11.307, ValidAcc : 51.92%
    # TrainLoss : 5.54299, TrainAcc: 80.66%, ValidLoss : 10.957, ValidAcc : 53.39%
    # TrainLoss : 5.38813, TrainAcc: 86.33%, ValidLoss : 11.890, ValidAcc : 48.88%
    # TrainLoss : 5.45038, TrainAcc: 82.32%, ValidLoss : 11.363, ValidAcc : 52.38%
    # TrainLoss : 5.39364, TrainAcc: 85.94%, ValidLoss : 11.910, ValidAcc : 51.63%
    # TrainLoss : 5.35891, TrainAcc: 86.04%, ValidLoss : 12.041, ValidAcc : 51.68%
    # TrainLoss : 5.28192, TrainAcc: 86.82%, ValidLoss : 12.052, ValidAcc : 52.27%
    # TrainLoss : 5.34827, TrainAcc: 85.74%, ValidLoss : 11.706, ValidAcc : 50.90%
    # TrainLoss : 5.20638, TrainAcc: 88.18%, ValidLoss : 11.707, ValidAcc : 52.35%
    # TrainLoss : 5.27451, TrainAcc: 85.16%, ValidLoss : 11.393, ValidAcc : 55.00%
    # TrainLoss : 5.21093, TrainAcc: 88.28%, ValidLoss : 12.389, ValidAcc : 52.37%
    # TrainLoss : 5.17348, TrainAcc: 88.96%, ValidLoss : 12.117, ValidAcc : 54.03%
    # TrainLoss : 5.19325, TrainAcc: 87.40%, ValidLoss : 12.123, ValidAcc : 51.53%
    # TrainLoss : 5.13403, TrainAcc: 88.38%, ValidLoss : 11.930, ValidAcc : 51.40%
    # TrainLoss : 5.19108, TrainAcc: 84.67%, ValidLoss : 12.322, ValidAcc : 50.95%
    # TrainLoss : 5.10859, TrainAcc: 86.82%, ValidLoss : 12.071, ValidAcc : 55.06%
    # TrainLoss : 5.05195, TrainAcc: 89.26%, ValidLoss : 12.422, ValidAcc : 53.48%
    # TrainLoss : 5.03893, TrainAcc: 88.18%, ValidLoss : 12.834, ValidAcc : 48.89%
    # TrainLoss : 5.01282, TrainAcc: 88.67%, ValidLoss : 12.496, ValidAcc : 52.38%
    # TrainLoss : 5.00897, TrainAcc: 88.57%, ValidLoss : 13.116, ValidAcc : 47.96%
    # TrainLoss : 5.01013, TrainAcc: 87.70%, ValidLoss : 12.908, ValidAcc : 49.58%
    # TrainLoss : 4.92019, TrainAcc: 91.41%, ValidLoss : 12.612, ValidAcc : 53.84%
    # TrainLoss : 4.92470, TrainAcc: 89.36%, ValidLoss : 12.598, ValidAcc : 53.05%
    # TrainLoss : 4.88102, TrainAcc: 90.23%, ValidLoss : 12.972, ValidAcc : 53.11%
    # TrainLoss : 4.84855, TrainAcc: 90.23%, ValidLoss : 12.454, ValidAcc : 52.63%
    # TrainLoss : 4.86205, TrainAcc: 91.60%, ValidLoss : 12.739, ValidAcc : 47.57%
    # TrainLoss : 4.83381, TrainAcc: 91.11%, ValidLoss : 13.004, ValidAcc : 53.10%
    # TrainLoss : 4.76931, TrainAcc: 92.29%, ValidLoss : 13.468, ValidAcc : 48.20%
    # TrainLoss : 4.79827, TrainAcc: 90.92%, ValidLoss : 12.810, ValidAcc : 51.71%
    # TrainLoss : 4.75536, TrainAcc: 91.02%, ValidLoss : 13.113, ValidAcc : 52.60%
    # TrainLoss : 4.78419, TrainAcc: 90.82%, ValidLoss : 13.527, ValidAcc : 49.24%
    # TrainLoss : 4.76852, TrainAcc: 90.92%, ValidLoss : 13.495, ValidAcc : 51.65%
    # TrainLoss : 4.70449, TrainAcc: 92.87%, ValidLoss : 13.026, ValidAcc : 52.22%
    # TrainLoss : 4.66103, TrainAcc: 93.07%, ValidLoss : 14.132, ValidAcc : 47.78%