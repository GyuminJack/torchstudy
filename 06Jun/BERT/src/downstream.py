import torch
import torch.nn as nn
from sklearn.metrics import f1_score
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


class bertclassifier(nn.Module):
    def __init__(self, bert, hid_dim, out_dim):
        super().__init__()
        self.bert = bert
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.relu = torch.nn.ReLU()
        self.fc_2 = nn.Linear(256, out_dim)

    def forward(self, src, src_mask, segment):
        # batch_size = src.shape[0]
        # src_len = src.shape[1]
        # pos = self.bert.pos_embedding(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device))
        # segment = self.bert.segment_embedding(segment)
        # src = self.bert.tok_embedding(src) + pos + segment

        # for layer in self.bert.layers:
        #     src = layer(src, src_mask)

        # self.bert.eval()
        src = self.bert.encode(src, src_mask, segment)
        out = self.relu(self.fc_1(src[:, 0, :]))
        out = self.fc_2(out)
        return out

def train(iterator, model, optimizer, scheduler, loss_fn,  device):
    model.to(device)
    ret_loss = 0
    for m_batch, (batch_dict, label) in enumerate(iterator):
        optimizer.zero_grad()

        input_tokens = batch_dict["input_ids"].to(device)
        segments = batch_dict["token_type_ids"].to(device)
        attention_masks = batch_dict["attention_mask"].unsqueeze(1).unsqueeze(2).to(device)
        label = label.to(device)

        output = model(input_tokens, attention_masks, segments)

        # flatten_output = output.reshape(-1, output.size(-1))
        ids = torch.argmax(output, dim=-1)
        mini_f1 = f1_score(label.reshape(-1).to('cpu'), ids.reshape(-1).to('cpu'), average='macro')

        total_loss = loss_fn(output, label)
        total_loss.backward()

        ret_loss += total_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print(f"{m_batch}/{len(iterator)}, {total_loss.item():.3f}, {mini_f1*100:.2f}", end="\r")
        
    epoch_loss = ret_loss / len(iterator)
    return epoch_loss

def evaluate_simple(iterator, model, loss_fn, device):
    model.eval()
    epoch_loss = 0
    model = model.to(device)
    acc = 0
    with torch.no_grad():
        step = 0
        f1 = 0
        for batch_dict, label in iterator:
            st = time.time()
            input_tokens = batch_dict["input_ids"].long().to(device)
            segments = batch_dict["token_type_ids"].long().to(device)
            attention_masks = batch_dict["attention_mask"].long().unsqueeze(1).unsqueeze(2).to(device)
            label = label.to(device)

            output = model(input_tokens, attention_masks, segments)
            loss = loss_fn(output, label)

            step_loss_val = loss.item()
            epoch_loss += step_loss_val

            ids = torch.argmax(output, dim=-1)
            mini_f1 = f1_score(label.reshape(-1).to('cpu'), ids.reshape(-1).to('cpu'), average='macro')
            hit = sum(list(label.reshape(-1).to('cpu') == ids.reshape(-1).to('cpu').flatten()))
            step += 1
            f1 += mini_f1
            acc += hit/len(list(label))
            # print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")

    return epoch_loss / len(iterator), acc/len(iterator)*100, f1/len(iterator)*100

if __name__ == "__main__":
    from data import *
    from torch.utils.data import DataLoader

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    txt_path = "/home/jack/torchstudy/05May/ELMo/data/ynat/train_tokenized.ynat"
    valid_txt_path = "/home/jack/torchstudy/05May/ELMo/data/ynat/val_tokenized.ynat"
    
    train_dataset = KoDataset_with_label_ynat(vocab_txt_path, txt_path)
    train_data_loader = DataLoader(train_dataset, collate_fn= train_dataset.collate_fn, batch_size=64)
    
    valid_dataset = KoDataset_with_label_ynat(vocab_txt_path, valid_txt_path)
    valid_data_loader = DataLoader(valid_dataset, collate_fn= valid_dataset.collate_fn, batch_size=256)

    bert = torch.load('../models/best_petition_namu_18.pt').module
    classifier = bertclassifier(bert, 768, 7)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr = 0.0005, weight_decay = 0.01)
    # scheduler = None
    scheduler = WarmupConstantSchedule(optimizer, 768, len(train_data_loader) * 5)

    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda:1'
    for i in range(50):
        train(train_data_loader, classifier, optimizer, scheduler, loss_fn,  device)
        loss, acc, f1 = evaluate_simple(valid_data_loader, classifier, loss_fn,  device)
        print(f"{loss:.3f} / {acc:.3f} / {f1:.3f}")
