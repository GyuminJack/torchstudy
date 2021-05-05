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
    for i, j in TrainDataloader:
        optimizer.zero_grad()
        input = j[:-1]
        trg = i[1:].reshape(-1,)
        pred = elmo(input)
        pred = pred.reshape(-1, len(TrainDataset.ko_vocab))
        loss = criterion(pred, trg)
        print(loss)
        loss.backward()
        optimizer.step()
    