from src.trainer import s2sTrainer
from src.data import DeEndataset
from torch.utils.data import DataLoader
import torch

def get_datasets():
    train_data_paths = [
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/train.de",
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/train.en"
    ]

    valid_data_paths = [
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/val.de",
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/val.en"
        ]

    test_data_paths = [
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/test2016.de",
        "/home/jack/torchstudy/2week/1_refcode/.data/multi30k/test2016.en"
        ]

    TrainDataset = DeEndataset(train_data_paths)

    # 파일로 로딩하기 때문에 각각의 오브젝트를 따로 만들고, vocab만 공유하는 방식으로 구성함.
    ValidDataset = DeEndataset(valid_data_paths)
    ValidDataset.src_vocab = TrainDataset.src_vocab
    ValidDataset.dst_vocab = TrainDataset.dst_vocab

    TestDataset = DeEndataset(test_data_paths)
    TestDataset.src_vocab = TestDataset.src_vocab
    TestDataset.dst_vocab = TestDataset.dst_vocab

    return TrainDataset, ValidDataset, TestDataset


if __name__ == "__main__" : 
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    TrainDataset, ValidDataset, TestDataset = get_datasets()
    
    BATCH_SIZE = 128
    TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    TestDataloader = DataLoader(TestDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
    
    encoder_config = {
        "emb_dim" : 1000,
        "hid_dim" : 1000,
        "lstm_layers" : 4,
        "num_embeddings" : len(TrainDataset.src_vocab),
        "pad_idx" : TrainDataset.src_vocab.pad_idx
    }
   
    decoder_config = {
        "emb_dim" : 1000,
        "hid_dim" : 1000,
        "lstm_layers" : 4,
        "num_embeddings" : len(TrainDataset.dst_vocab),
        "pad_idx" : TrainDataset.dst_vocab.pad_idx,
        "output_dim" : len(TrainDataset.dst_vocab)
    }

    epoch = 200

    trainer = s2sTrainer(encoder_config, decoder_config, device)
    trainer.run(epoch, TrainDataloader, ValidDataloader)