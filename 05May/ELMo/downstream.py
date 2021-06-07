import torch 
import dill
from konlpy.tag import Mecab
from src.data import *
from src.models import *
from src.downstream import GRUClassifier, BaseGRUClassifier
import time
import numpy as np
from sklearn.metrics import f1_score
import mlflow

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

def read_pkl(path):
    with open(path, "rb") as f:
        return dill.load(f)

def train(model, iterator, optimizer, criterion, clip, device, scheduler=None):
    model.train()
    epoch_loss = 0
    step = 0
    acc = 0
    f1 = 0
    for padded_sentence, padded_char, labels in iterator:
        tot = 0
        hit = 0
        tot += len(labels)
        optimizer.zero_grad()

        padded_sentence = padded_sentence
        padded_char = padded_char
        labels = labels.to(device)
        st = time.time()
        step += 1

        output = model([padded_char, padded_sentence])
        loss = criterion(output, labels)
        loss.backward()

        ids = torch.argmax(output, dim=-1)
        hit = (ids.reshape(-1)==labels.reshape(-1)).sum().item()
        f1 += f1_score(labels.reshape(-1).to('cpu'), ids.reshape(-1).to('cpu'), average='macro')
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        step_loss_val = loss.item()
        epoch_loss += step_loss_val
        acc += hit/tot
        # print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%), step_acc : {hit/tot*100:.2f}, time : {time.time()-st:.3f}s", end="\r")
    return epoch_loss / len(iterator), acc / len(iterator) * 100, f1/len(iterator)

def evaluate(model, criterion, iterator, id2word, id2label, epoch, device):
    model.eval()
    f = open(f"./eval/val/ynat/valid_{epoch}.txt", "w")
    epoch_loss = 0

    acc = 0
    with torch.no_grad():
        step = 0
        f1 = 0
        for padded_sentence, padded_char, labels in iterator:
            tot = 0
            hit = 0
            tot += len(labels)
            labels = labels.to(device)
            output = model([padded_char, padded_sentence])
            loss = criterion(output, labels)
            step_loss_val = loss.item()
            epoch_loss += step_loss_val
            

            ids = torch.argmax(output, dim=-1)
            mini_f1 = f1_score(labels.reshape(-1).to('cpu'), ids.reshape(-1).to('cpu'), average='macro')
            for i, sentence in enumerate(padded_sentence.T):
                try:
                    orig_label = id2label[int(labels[i])]
                except:
                    orig_label = f'기타({int(labels[i])})' 
                pred_label = id2label[int(ids[i])]
                if orig_label == pred_label:
                    hit += 1
                f.write("".join([orig_label, "||" ,pred_label, "||", " ".join([id2word[int(j)]for j in sentence if int(j) not in [0,2,3]]),"\n"]))
            step += 1
            f1 += mini_f1
            acc += hit/tot
            # print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")
    f.close()
    return epoch_loss / len(iterator), acc/len(iterator)*100, f1/len(iterator)

def test(model, criterion, iterator, id2word, id2label, epoch, device):
    model.eval()
    f = open(f"./eval/test/ynat/test_{epoch}.txt", "w")
    epoch_loss = 0
    with torch.no_grad():
        step = 0
        for padded_sentence, padded_char, labels in iterator:
            labels = labels
            output = model([padded_char, padded_sentence])
            ids = torch.argmax(output, dim=-1)
            for i, sentence in enumerate(padded_sentence.T):
                try:
                    orig_label = id2label[int(labels[i])]
                except:
                    orig_label = f'기타({int(labels[i])})' 
                pred_label = id2label[int(ids[i])]
                f.write("".join([orig_label, "||" ,pred_label, "||", " ".join([id2word[int(j)]for j in sentence if int(j) not in [0,2,3]]),"\n"]))
    f.close()

def get_petitions(elmo_dataset):

    TrainDataset = KoDownStreamDataset(
                                "./data/downstream/petitions.downstream.ko.patch1.train", 
                                elmo_dataset.ko_vocab, 
                                elmo_dataset.character_dict, 
                                elmo_dataset.max_character_length
                            )
    
    ValidDataset = KoDownStreamDataset(
                            "./data/downstream/petitions.downstream.ko.patch1.valid", 
                            elmo_dataset.ko_vocab, 
                            elmo_dataset.character_dict, 
                            elmo_dataset.max_character_length
                        )
    TestDataset = KoDownStreamDataset(
                            "./data/downstream/petitions.downstream.ko.patch1.test", 
                            elmo_dataset.ko_vocab, 
                            elmo_dataset.character_dict, 
                            elmo_dataset.max_character_length
                       )

def get_ynat(elmo_dataset):

    TrainDataset = YnatDownStreamDataset(
                                "./data/ynat/train_tokenized.ynat", 
                                elmo_dataset.ko_vocab, 
                                elmo_dataset.character_dict, 
                                elmo_dataset.max_character_length
                            )
    
    ValidDataset = YnatDownStreamDataset(
                            "./data/ynat/val_tokenized.ynat", 
                            elmo_dataset.ko_vocab, 
                            elmo_dataset.character_dict, 
                            elmo_dataset.max_character_length
                        )
    TestDataset = YnatDownStreamDataset(
                            "./data/ynat/test_tokenized.ynat", 
                            elmo_dataset.ko_vocab, 
                            elmo_dataset.character_dict, 
                            elmo_dataset.max_character_length
                       )
                        
                    
    ValidDataset.label_dict = TrainDataset.label_dict
    TestDataset.label_dict = TrainDataset.label_dict
    return TrainDataset, ValidDataset, TestDataset

def get_model(elmo, elmo_dataset, TrainDataset, model_type, elmo_w2v = True, w2v_init = True, device = "cuda:1"):
    gru_emb_dim = 200
    
    if model_type == 'elmo':
        if elmo_w2v:
            model_type = "elmo_with_w2v"
            model = GRUClassifier(elmo = elmo, 
                                elmo_train_vocab = elmo_dataset.ko_vocab,
                                elmo_hidden_output_dim = 512,  # elmo lstm hidden size * 2 (bidirectional)
                                elmo_projection_dim = -1,      # projection dimension, if -1 no projection
                                emb_dim = gru_emb_dim,                  # independent Embedding size
                                hid_dim = 128,                  # GRU hidden size
                                output_dim = len(TrainDataset.label_dict),                  # for classification (many to one)
                                w2v_path = "./vocab/w2v/w2v_all_news_200_epoch19.model",    # for word2vec initialize
                                n_layers = 2,                   # GRU n_layers
                                device = device
                                )
            if w2v_init == False:
                model_type = "elmo_with_random"
                model = GRUClassifier(elmo = elmo, 
                                    elmo_train_vocab = elmo_dataset.ko_vocab,
                                    elmo_hidden_output_dim = 512,  
                                    elmo_projection_dim = -1,      
                                    emb_dim = gru_emb_dim,                  
                                    hid_dim = 512,                  
                                    output_dim = len(TrainDataset.label_dict),         
                                    w2v_path = "random",    
                                    n_layers = 2,               
                                    device = device
                                    )                
        else:
            model_type = "elmo_without_w2v"
            model = GRUClassifier(elmo = elmo, 
                                elmo_train_vocab = elmo_dataset.ko_vocab,
                                elmo_hidden_output_dim = 512, 
                                elmo_projection_dim = -1, 
                                use_idp_emb = False,
                                hid_dim = 512, 
                                output_dim = len(TrainDataset.label_dict), 
                                w2v_path = None,
                                n_layers = 2, 
                                device = device
                                )

        for name, param in model.named_parameters():
            if 'elmo' in name:
                param.requires_grad = False
    
    # not use ELMo (Baseline)
    elif model_type == "only_w2v_init":
        # use Word2Vec model
        model_type = "only_w2v_init"
        model = BaseGRUClassifier(
                            vocab = elmo_dataset.ko_vocab,
                            emb_dim = gru_emb_dim,
                            hid_dim = 512,
                            output_dim = len(TrainDataset.label_dict),
                            w2v_path =  "./vocab/w2v/w2v_all_news_200_epoch19.model",
                            n_layers = 2, 
                            dropout_rate = 0.5, 
                            pad_idx = 0,
                            device = device
                        )
        
    elif model_type == "random_emb_init":
        # not using Word2Vec model
        model_type = "random_emb_init"
        model = BaseGRUClassifier(
                        vocab = elmo_dataset.ko_vocab,
                        emb_dim = gru_emb_dim,
                        hid_dim = 512,
                        output_dim = len(TrainDataset.label_dict),
                        w2v_path = None,
                        n_layers = 2, 
                        dropout_rate = 0.5, 
                        pad_idx = 0,
                        device = device
                        )

    model = model.to(device)
    return model_type, model

if __name__ == "__main__":
    
    vocab_path = "./vocab/petition_ynat.pkl"
    elmo_dataset = read_pkl(vocab_path)
    TrainDataset, ValidDataset, TestDataset = get_ynat(elmo_dataset)
    id2label_dict = {v : k for k, v in TrainDataset.label_dict.items()}

    # ----------- Params Setting --------------
    device = "cuda:0"
    model_type = 'elmo' # [only_w2v_init, random_emb_init, elmo]
    elmo_w2v = True   # when model_type is elmo, option for using w2v
    w2v_init = False
    
    lr = 0.00005
    schedule = False
    d_model = 1024
    warmup = 500
    
    batch_size = 256
    elmo_path = "./model/petition_ynat_0531.pt"
    # ----------- Params Setting --------------

    elmo = torch.load(elmo_path, map_location=device)
    model_type, model = get_model(elmo, elmo_dataset, TrainDataset, model_type, elmo_w2v, w2v_init, device)

    TrainDataloader = DataLoader(TrainDataset, batch_size = batch_size, shuffle=True, collate_fn=TrainDataset.collate_fn)
    ValidDataloader = DataLoader(ValidDataset, batch_size = 30, shuffle=False, collate_fn=ValidDataset.collate_fn)
    TestDataloader = DataLoader(TestDataset, batch_size = 30, shuffle=False, collate_fn=TestDataset.collate_fn)
    
    criterion = nn.CrossEntropyLoss()

    if schedule:
        lr = -1
        optimizer = torch.optim.Adam(model.parameters(), lr = 1,  betas = (0.9, 0.98), eps=10e-9)
        scheduler = WarmupConstantSchedule(optimizer, d_model = d_model, warmup_steps = warmup)
    else:
        d_model = -1
        warmup = -1
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        scheduler = None

    params = {
        'elmo_path' : elmo_path,
        'vocab_path' : vocab_path,
        "model_type" : model_type,
        "elmo_w2v" : elmo_w2v,
        "w2v_init" : w2v_init,
        "schedule" : schedule,
        "dmodel_warmupstep" : "_".join([str(d_model), str(warmup)]),
        "lr" : lr,
        "batch_size" : batch_size
    }

    _train = True

    if _train == True:
        
        with mlflow.start_run(experiment_id = '1', run_name=model_type) as run:
            mlflow.log_params(params)
            p_valid_acc, p_valid_f1 = 0, 0

            for epoch in range(200):
                st = time.time()
                train_loss, train_acc, train_f1 = train(model, TrainDataloader, optimizer, criterion, clip = 2, device = device, scheduler = scheduler)
                eval_loss, valid_acc, valid_f1 = evaluate(model, criterion, ValidDataloader, TrainDataset.id2word_dict, id2label_dict, model_type + str(epoch), device = device)

                mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                mlflow.log_metric(key="train_acc", value=train_acc, step=epoch)
                mlflow.log_metric(key="train_f1", value=train_f1, step=epoch)
                mlflow.log_metric(key="eval_loss", value=eval_loss, step=epoch)
                mlflow.log_metric(key="valid_acc", value=valid_acc, step=epoch)
                mlflow.log_metric(key="valid_f1", value=valid_f1, step=epoch)

                if valid_acc > p_valid_acc:
                    p_valid_acc = valid_acc
                    mlflow.log_metric(key='max_valid_acc', value=valid_acc)
                
                if valid_f1 > p_valid_f1:
                    p_valid_f1 = valid_f1
                    mlflow.log_metric(key='max_valid_f1', value=valid_f1)
                
                # print(f"train_loss : {train_loss:.3f}, train_acc : {train_acc:.3f}, train_f1 : {train_f1:.3f}   valid_loss : {eval_loss:.3f}, valid_acc : {valid_acc:.3f},  valid_f1 : {valid_f1:.3f}  time : {time.time()-st:.3f}s", flush = True)
                if (epoch+1) % 10 == 0:
                    test(model, criterion, TestDataloader, TrainDataset.id2word_dict, id2label_dict, model_type + str(epoch), device = device)
                    # mlflow.pytorch.log_model(model, artifact_path="model")
                    # model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
                    # mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
                    torch.save(model, f"./model/ynat/{model_type+str(epoch+1)}.pt")
            mlflow.log_param('last_model_save', f"./model/ynat/{model_type+str(epoch+1)}.pt")

    elif _train == False:
        model = torch.load("./model/tmp.pt")
        evaluate(model, criterion, ValidDataloader, TrainDataset.id2word_dict, id2label_dict, 'custom_test', device = device) 