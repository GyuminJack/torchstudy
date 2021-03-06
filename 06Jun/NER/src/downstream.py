import torch
import torch.nn as nn
from torchcrf import CRF
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

class BertNER(nn.Module):
    def __init__(self, bert, bert_hidden_size, num_classes):
        super().__init__()
        self.bert = bert
        self.position_wise_ff = nn.Linear(bert_hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, src, src_mask, segment, tags):
        src = self.bert.encode(src, src_mask, segment)
        last_encoder_layer = self.dropout(src)
        emissions = self.position_wise_ff(last_encoder_layer)
        log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
        return log_likelihood, sequence_of_tags


def train(iterator, model, optimizer, scheduler,  device):
    model.to(device)
    ret_loss = 0
    _acc = 0
    for m_batch, (batch_dict, label) in enumerate(iterator):
        optimizer.zero_grad()

        input_tokens = batch_dict["input_ids"].to(device)
        segments = batch_dict["token_type_ids"].to(device)
        attention_masks = batch_dict["attention_mask"].unsqueeze(1).unsqueeze(2).to(device)
        label = label.to(device)

        log_likelihood, sequence_of_tags = model(input_tokens, attention_masks, segments, label)
        # flatten_output = output.reshape(-1, output.size(-1))
        loss = -1 * log_likelihood
        
        # if n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss.backward()
        ret_loss += loss.item()

        _acc = ((label.reshape(-1).to('cpu')==torch.Tensor(sequence_of_tags).reshape(-1).to('cpu')).float().sum()/len(label.reshape(-1)))
        
        # Average of F1-score without 'O' tag
        _l = label.reshape(-1).to('cpu')
        _st = torch.Tensor(sequence_of_tags).reshape(-1)
        _f1 = 0
        for i in range(1, 13):
            if sum((_l == i).long()) == 0:
                _f1 += 0
            else:
                _f1 += f1_score((_l == i).long(), (_st == i).long())
        mini_f1 = _f1/12

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print(f"{m_batch}/{len(iterator)}, {loss.item():.3f}, {_acc*100:.2f}, {mini_f1*100:.2f}", end="\r")

    acc = _acc / len(iterator)
    epoch_loss = ret_loss / len(iterator)
    return epoch_loss, acc * 100

def evaluate_simple(iterator, model, device):
    model.eval()
    epoch_loss = 0
    model = model.to(device)
    acc = 0
    f1 = 0
    with torch.no_grad():
        step = 0
        f1 = 0
        for batch_dict, label in iterator:
            st = time.time()
            input_tokens = batch_dict["input_ids"].long().to(device)
            segments = batch_dict["token_type_ids"].long().to(device)
            attention_masks = batch_dict["attention_mask"].long().unsqueeze(1).unsqueeze(2).to(device)
            label = label.to(device)

            log_likelihood, sequence_of_tags = model(input_tokens, attention_masks, segments, label)
            loss = -1 * log_likelihood

            step_loss_val = loss.item()
            epoch_loss += step_loss_val

            acc += ((label.reshape(-1).to('cpu')==torch.Tensor(sequence_of_tags).reshape(-1).to('cpu')).float().sum()/len(label.reshape(-1)))

            _l = label.reshape(-1).to('cpu')
            _st = torch.Tensor(sequence_of_tags).reshape(-1)
            _f1 = 0
            for i in range(1, 13):
                if sum((_l == i).long()) == 0:
                    _f1 += 0
                else:
                    _f1 += f1_score((_l == i).long(), (_st == i).long())
            mini_f1 = _f1/12
            f1 += mini_f1
            step += 1
            
    return epoch_loss / len(iterator), acc/len(iterator)*100, f1/len(iterator)*100



if __name__ == "__main__":
    from data import *
    from torch.utils.data import DataLoader

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"

    train_path = "/home/jack/torchstudy/06Jun/NER/data/namu_2021060809/klue_ner_20210715.train"
    train_dataset = KlueDataset_NER(vocab_txt_path, train_path)
    train_data_loader = DataLoader(train_dataset, collate_fn= lambda batch: train_dataset.collate_fn(batch), batch_size=128, shuffle=True)
    
    valid_path = "/home/jack/torchstudy/06Jun/NER/data/namu_2021060809/klue_ner_20210715.dev"
    valid_dataset = KlueDataset_NER(vocab_txt_path, valid_path)
    valid_data_loader = DataLoader(valid_dataset, collate_fn= lambda batch: valid_dataset.collate_fn(batch), batch_size=256)

    bert = torch.load('../../BERT/models/best_petition_namu_17.pt').module
    classifier = BertNER(bert, 768, len(train_dataset.bio_dict))

    optimizer = torch.optim.AdamW(classifier.parameters(), lr = 0.0001, weight_decay = 0.01)
    # scheduler = None
    scheduler = WarmupConstantSchedule(optimizer, 768, 2000)

    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda:1'
    for i in range(50):
        train(train_data_loader, classifier, optimizer, scheduler,  device)
        loss, acc, f1 = evaluate_simple(valid_data_loader, classifier,  device)
        print(f"{loss:.3f} / {acc:.3f} / {f1:.3f}                 ")
