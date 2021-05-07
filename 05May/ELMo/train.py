from src.data import *
from src.models import *
import torch.optim as optim

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    TrainDataset = KoDataset("/home/jack/torchstudy/05May/ELMo/data/train.ko")

    TrainDataloader = DataLoader(TrainDataset, batch_size = 2, shuffle=True, collate_fn=TrainDataset.collate_fn)

    emb_dim = 256
    ch = CnnHighway(128, emb_dim, TrainDataset.character_dict)
    elmo = ELMo(ch, emb_dim, 1024, len(TrainDataset.ko_vocab))

    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.Adam(elmo.parameters())

    for original, char_input in TrainDataloader:
        optimizer.zero_grad()

        elmo_input = char_input[:,:-1,:]
        original_trg = original[:,1:]

        fpred, bpred = elmo(elmo_input)
        forward_loss = criterion(fpred.reshape(-1, len(TrainDataset.ko_vocab)), original_trg.reshape(-1))
        backward_loss = criterion(bpred.reshape(-1, len(TrainDataset.ko_vocab)), original_trg.reshape(-1))
        loss = forward_loss + backward_loss
        loss.backward()
        optimizer.step()