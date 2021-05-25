from src.data import *
from src.models import *
import torch.optim as optim
from src.trainer import *
from torch.utils.data import DataLoader
import time
import dill

def save_pkl(obj, obj_path):
    with open(obj_path, "wb") as f:
        dill.dump(obj, f)

if __name__ == "__main__":
    st = time.time()
    TrainDataset = KoDataset("/home/jack/torchstudy/05May/ELMo/data/new_cleaned_petition.ko.patch1", 
                max_character_length = 5,
                max_character_size = 800, # FIXED VALUE 
                max_vocab_size = 40000
                )
    
    save_pkl(TrainDataset, "./vocab/traindataset.pkl")

    print(f"1. Finish TrainDataset({time.time()-st:.3f}s)")

    emb_layer = 256
    model_config = {
        "character_embedding" : 128,
        'cnn_kernal_output' : [[2, int(emb_layer/2)],[3, int(emb_layer/2)]],
        "word_embedding" : emb_layer, # sum of cnn_kernal_output 
        "character_set_size" : len(TrainDataset.character_dict),
        'highway_layer' : 2,
        "hidden_size" : emb_layer,
        "output_dim" : len(TrainDataset.ko_vocab)
    }

    train_config = {
        "epochs" : 400,
        "device" : "cuda:1",
        "batch_size" : 96,
        "lr" : 0.001,
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