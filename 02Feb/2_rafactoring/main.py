from src.trainer import s2sTrainer
from src.data import DeEndataset, KoEndataset
from torch.utils.data import DataLoader
import torch
import sys
from src.seq2seq import Encoder, Decoder, seq2seq

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


def get_en_ko_datesets():

    train_data_paths = [
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.train.ko",
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.train.en"
    ]

    valid_data_paths = [
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.dev.ko",
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.dev.en"
        ]

    test_data_paths = [
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.test.ko",
        "/home/jack/torchstudy/02Feb/0_datas/korean-english-park.test.en"
        ]

    TrainDataset = KoEndataset(train_data_paths)

    ValidDataset = KoEndataset(valid_data_paths)
    ValidDataset.src_vocab = TrainDataset.src_vocab
    ValidDataset.dst_vocab = TrainDataset.dst_vocab
    
    TestDataset = KoEndataset(test_data_paths)
    TestDataset.src_vocab = TestDataset.src_vocab
    TestDataset.dst_vocab = TestDataset.dst_vocab

    return TrainDataset, ValidDataset, TestDataset



if __name__ == "__main__" : 
    if sys.argv[1] == "train":
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        TrainDataset, ValidDataset, TestDataset = get_en_ko_datesets()
        
        BATCH_SIZE = 128
        TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
        ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
        TestDataloader = DataLoader(TestDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=DeEndataset.collate_fn)
        
        encoder_config = {
            "emb_dim" : 1000,
            "hid_dim" : 1000,
            "n_layers" : 4,
            "input_dim" : len(TrainDataset.src_vocab),
            "pad_idx" : TrainDataset.src_vocab.pad_idx
        }
    
        decoder_config = {
            "emb_dim" : 1000,
            "hid_dim" : 1000,
            "n_layers" : 4,
            "pad_idx" : TrainDataset.dst_vocab.pad_idx,
            "output_dim" : len(TrainDataset.dst_vocab)
        }

        epoch = 100

        trainer = s2sTrainer(encoder_config, decoder_config, device)
        trainer.run(epoch, TrainDataloader, ValidDataloader)
    
    elif sys.argv[1] == "live":
        TrainDataset, _, _ = get_en_ko_datesets()

        encoder_config = {
            "emb_dim" : 1000,
            "hid_dim" : 1000,
            "n_layers" : 4,
            "input_dim" : len(TrainDataset.src_vocab),
            "pad_idx" : TrainDataset.src_vocab.pad_idx
        }
    
        decoder_config = {
            "emb_dim" : 1000,
            "hid_dim" : 1000,
            "n_layers" : 4,
            "pad_idx" : TrainDataset.dst_vocab.pad_idx,
            "output_dim" : len(TrainDataset.dst_vocab)
        }

        encoder = Encoder(**encoder_config)
        decoder = Decoder(**decoder_config)
        seq2seq = seq2seq(encoder, decoder)
        seq2seq.load_state_dict(torch.load("./seq2seq-model.pt"), strict=False)
        TrainDataset.dst_vocab.build_index_dict()
        while True:
            src = input("Input : ")
            test_sample = torch.Tensor([TrainDataset.src_vocab.stoi(src.lower(), option="seq2seq", reverse=True)]).long()
            print(test_sample)
            test_sample = torch.reshape(test_sample, (-1,1))
            pred, pred_probs = seq2seq.predict(test_sample)
            print(pred)
            print(TrainDataset.dst_vocab.itos(pred))

            