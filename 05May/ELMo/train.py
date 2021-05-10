from src.data import *
from src.models import *
import torch.optim as optim
from src.trainer import *

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    TrainDataset = KoDataset("/home/jack/torchstudy/05May/ELMo/data/train.ko")

    TrainDataloader = DataLoader(TrainDataset, batch_size = 2, shuffle=True, collate_fn=TrainDataset.collate_fn)

    model_config = {
        "cnn_embedding" :128,
        "emb_dim" : 256,
        "character_dict" : TrainDataset.character_dict,
        "hidden_size" : 1024,
        "output_dim" : len(TrainDataset.ko_vocab)
    }

    train_config = {
        "epochs" : 100,
        "device" : "cuda:1"
    }
    
    train_config.update(model_config)
    trainer = Trainer(train_config)

    trainer.run(TrainDataloader)