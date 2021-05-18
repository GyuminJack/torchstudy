from src.data import *
from src.models import *
import torch.optim as optim
from src.trainer import *
from torch.utils.data import DataLoader
import time
if __name__ == "__main__":
    st = time.time()
    TrainDataset = KoDataset("/home/jack/torchstudy/05May/ELMo/data/new_cleaned_petition.ko", 
                max_character_length = 5,
                max_character_size = 800, # FIXED VALUE 
                max_vocab_size = 30000
                )
    
    print(f"1. Finish TrainDataset({time.time()-st:.3f}s)")


    model_config = {
        "character_embedding" : 128,
        'cnn_kernal_output' : [[2, 256],[3, 256],[4, 256]],
        "word_embedding" : 256,
        "character_set_size" : len(TrainDataset.character_dict),
        'highway_layer' : 2,
        "hidden_size" : 512,
        "output_dim" : len(TrainDataset.ko_vocab)
    }

    train_config = {
        "epochs" : 150,
        "device" : "cuda:0",
        "batch_size" : 96,
        "lr" : 0.005,
        "optimizer" : torch.optim.Adam,
        "schedule" : False
    }
    
    TrainDataloader = DataLoader(TrainDataset, batch_size = train_config['batch_size'], shuffle=True, collate_fn=TrainDataset.collate_fn)
    print("2. Finish call DataLoader")

    import pprint
    pprint.pprint(model_config)
    pprint.pprint(train_config)

    print("3. Start Train")
    train_config.update(model_config)
    
    trainer = Trainer(train_config)
    trainer.run(TrainDataloader)