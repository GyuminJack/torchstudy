from src.data import *
from src.models import *
import torch.optim as optim
from src.trainer import *
from torch.utils.data import DataLoader
import time
if __name__ == "__main__":
    # 1. Finish TrainDataset(27.064s)
    # 2. Finish call DataLoader
    # 3. Start Train
    # Epoch: 01 | Time: 9m 23s | Train Loss: 12.801 | Train PPL: 362483.950
    # Epoch: 02 | Time: 9m 28s | Train Loss: 12.720 | Train PPL: 334530.903
    # Epoch: 03 | Time: 9m 29s | Train Loss: 12.712 | Train PPL: 331730.091
    # Epoch: 04 | Time: 9m 33s | Train Loss: 12.709 | Train PPL: 330850.654
    # Epoch: 05 | Time: 9m 31s | Train Loss: 12.708 | Train PPL: 330281.216
    # Epoch: 06 | Time: 9m 32s | Train Loss: 12.707 | Train PPL: 329984.478
    # Epoch: 07 | Time: 9m 30s | Train Loss: 12.706 | Train PPL: 329754.550
    # Epoch: 08 | Time: 9m 29s | Train Loss: 12.705 | Train PPL: 329550.406
    # Epoch: 09 | Time: 9m 30s | Train Loss: 12.705 | Train PPL: 329334.942
    # Epoch: 10 | Time: 9m 29s | Train Loss: 12.705 | Train PPL: 329289.986
    st = time.time()
    TrainDataset = KoDataset("/home/jack/torchstudy/05May/ELMo/data/300k_train_samples.ko", 
                max_character_length = 5,
                max_character_size = 1000,
                max_vocab_size = 10000
                )
    print(f"1. Finish TrainDataset({time.time()-st:.3f}s)")


    model_config = {
        "character_embedding" :256,
        'cnn_kernal_output' : [[2, 10], [3, 10], [4, 10]],
        "word_embedding" : 256,
        "character_set_size" : len(TrainDataset.character_dict),
        'highway_layer' : 1,
        "hidden_size" : 512,
        "output_dim" : len(TrainDataset.ko_vocab)
    }

    train_config = {
        "epochs" : 100,
        "device" : "cuda:1",
        "batch_size" : 256
    }

    TrainDataloader = DataLoader(TrainDataset, batch_size = train_config['batch_size'], shuffle=True, collate_fn=TrainDataset.collate_fn)
    print("2. Finish call DataLoader")


    print("3. Start Train")

    train_config.update(model_config)
    trainer = Trainer(train_config)
    trainer.run(TrainDataloader)