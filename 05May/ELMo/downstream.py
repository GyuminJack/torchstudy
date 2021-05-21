import torch 
import dill
from konlpy.tag import Mecab
from src.data import *
from src.models import *
from src.downstream import GRUClassifier
import time

def read_pkl(path):
    with open(path, "rb") as f:
        return dill.load(f)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    epoch_loss = 0
    step = 0
    for padded_sentence, padded_char, labels in iterator:
        st = time.time()
        step += 1
        optimizer.zero_grad()
        output = model([padded_char, padded_sentence])
        loss = criterion(output, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step_loss_val = loss.item()
        epoch_loss += step_loss_val

        print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")
    return epoch_loss / len(iterator)

if __name__ == "__main__":

    elmo_dataset = read_pkl("./vocab/traindataset.pkl")
    elmo = torch.load("./model/best_model_0521_4.pt", map_location="cpu")

    TrainDataset = KoDownStreamDataset(
                                "./data/petitions.downstream.ko.patch1", 
                                elmo_dataset.ko_vocab, 
                                elmo_dataset.character_dict, 
                                elmo_dataset.max_character_length
                            )

    TrainDataloader = DataLoader(TrainDataset, batch_size = 2, shuffle=True, collate_fn=TrainDataset.collate_fn)

    model = GRUClassifier(elmo = elmo, 
                        elmo_train_vocab = elmo_dataset.ko_vocab,
                        elmo_hidden_output_dim = 1024, 
                        elmo_projection_dim = 200, 
                        emb_dim = 200, 
                        hid_dim = 512, 
                        output_dim = len(TrainDataset.label_dict), 
                        w2v_path = "./vocab/w2v/w2v_all_news_200_epoch19.model",
                        n_layers = 2, 
                        dropout_rate = 0.5, 
                        pad_idx = 0
                        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, betas = (0.9, 0.98), eps=10e-9)

    train(model, TrainDataloader, optimizer, criterion, clip = 5)

    




