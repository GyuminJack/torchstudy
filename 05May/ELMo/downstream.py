import torch 
import dill
from konlpy.tag import Mecab
from src.data import *
from src.models import *
from src.downstream import GRUClassifier, BaseGRUClassifier
import time

def read_pkl(path):
    with open(path, "rb") as f:
        return dill.load(f)

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    step = 0
    for padded_sentence, padded_char, labels in iterator:
        optimizer.zero_grad()

        padded_sentence = padded_sentence
        padded_char = padded_char
        labels = labels.to(device)
        st = time.time()
        step += 1

        output = model([padded_char, padded_sentence])
        loss = criterion(output, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step_loss_val = loss.item()
        epoch_loss += step_loss_val

        # print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")
    return epoch_loss / len(iterator)

def evaluate(model, criterion, iterator, id2word, id2label, epoch, device):
    model.eval()
    f = open(f"./eval/val/valid_{epoch}.txt", "w")
    epoch_loss = 0
    with torch.no_grad():
        step = 0
        for padded_sentence, padded_char, labels in iterator:
            
            labels = labels.to(device)
            output = model([padded_char, padded_sentence])
            loss = criterion(output, labels)
            step_loss_val = loss.item()
            epoch_loss += step_loss_val

            if step < 20:
                ids = torch.argmax(output, dim=-1)
                for i, sentence in enumerate(padded_sentence.T):
                    try:
                        orig_label = id2label[int(labels[i])]
                    except:
                        orig_label = f'기타({int(labels[i])})' 
                    pred_label = id2label[int(ids[i])]
                    f.write("".join([orig_label, "||" ,pred_label, "||", " ".join([id2word[int(j)]for j in sentence if int(j) not in [0,2,3]]),"\n"]))
            step += 1
            # print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")
    f.close()
    return epoch_loss / len(iterator)

def test(model, criterion, iterator, id2word, id2label, epoch, device):
    model.eval()
    f = open(f"./eval/test/test_{epoch}.txt", "w")
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


if __name__ == "__main__":
    elmo_dataset = read_pkl("./vocab/traindataset.pkl")

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
                    
    ValidDataset.label_dict = TrainDataset.label_dict
    TestDataset.label_dict = TrainDataset.label_dict
    # print(TrainDataset.label_dict)

    TrainDataloader = DataLoader(TrainDataset, batch_size = 256, shuffle=True, collate_fn=TrainDataset.collate_fn)
    ValidDataloader = DataLoader(ValidDataset, batch_size = 30, shuffle=False, collate_fn=ValidDataset.collate_fn)
    TestDataloader = DataLoader(TestDataset, batch_size = 30, shuffle=False, collate_fn=TestDataset.collate_fn)

    id2label_dict = {v : k for k, v in TrainDataset.label_dict.items()}

    device = "cuda:0"
    model_type = 'random_emb_init' # [base, base_without_w2v, elmo]
    elmo_w2v = True   # when model_type is elmo, option for using w2v
    w2v_init = False
    ### Todo
    # 4가지 모델 로스 보기 , 베스트는 only w2v

    if model_type == 'elmo':
        elmo = torch.load("./model/best_model_0524_256.pt", map_location=device)
        if elmo_w2v:
            model_type = "elmo_with_w2v"
            model = GRUClassifier(elmo = elmo, 
                                elmo_train_vocab = elmo_dataset.ko_vocab,
                                elmo_hidden_output_dim = 512,  # elmo lstm hidden size * 2 (bidirectional)
                                elmo_projection_dim = -1,      # projection dimension, if -1 no projection
                                emb_dim = 200,                  # independent Embedding size
                                hid_dim = 512,                  # GRU hidden size
                                output_dim = len(TrainDataset.label_dict),                  # for classification (many to one)
                                w2v_path = "./vocab/w2v/w2v_all_news_200_epoch19.model",    # for word2vec initialize
                                n_layers = 2,                   # GRU n_layers
                                device = device
                                )
            if w2v_init == False:
                model_type = "elmo_with_random"
                model = GRUClassifier(elmo = elmo, 
                                    elmo_train_vocab = elmo_dataset.ko_vocab,
                                    elmo_hidden_output_dim = 512,  # elmo lstm hidden size * 2 (bidirectional)
                                    elmo_projection_dim = -1,      # projection dimension, if -1 no projection
                                    emb_dim = 200,                  # independent Embedding size
                                    hid_dim = 512,                  # GRU hidden size
                                    output_dim = len(TrainDataset.label_dict),                  # for classification (many to one)
                                    w2v_path = "random",    # for word2vec initialize
                                    n_layers = 2,                   # GRU n_layers
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
                            emb_dim = 200,
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
                        emb_dim = 200,
                        hid_dim = 512,
                        output_dim = len(TrainDataset.label_dict),
                        w2v_path = None,
                        n_layers = 2, 
                        dropout_rate = 0.5, 
                        pad_idx = 0,
                        device = device
                        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    
    _train = True

    if _train == True:
        for epoch in range(20):
            st = time.time()
            train_loss = train(model, TrainDataloader, optimizer, criterion, clip = 5, device = device)
            eval_loss = evaluate(model, criterion, ValidDataloader, TrainDataset.id2word_dict, id2label_dict, model_type + str(epoch), device = device)
            print(f"train_loss : {train_loss:.3f}, valid_loss : {eval_loss:.3f}, time : {time.time()-st:.3f}s", flush = True)
            if (epoch+1) % 10 == 0:
                test(model, criterion, TestDataloader, TrainDataset.id2word_dict, id2label_dict, model_type + str(epoch), device = device)
                torch.save(model, f"./model/{model_type+str(epoch+1)}.pt")

    elif _train == False:
        model = torch.load("./model/tmp.pt")
        evaluate(model, criterion, ValidDataloader, TrainDataset.id2word_dict, id2label_dict, 'custom_test', device = device)
    
    




