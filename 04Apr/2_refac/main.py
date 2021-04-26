from src.Trainer import *
from src.data import *
from infer import *
from src.model import *
import dill # for save
import os
import sys
import copy

def save_vocab(save_path, obj):
    with open(save_path, 'wb') as f:
        dill.dump(obj, f)

def read_vocab(save_path):
    with open(save_path, 'rb') as f:
        a = dill.load(f)
    return a

def train():
    
    BATCH_SIZE = 32

    train_data_paths = [
        "./data/src.tr",
        "./data/dst.tr"
    ]
    TrainDataset = KoKodataset(train_data_paths)
    TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=TrainDataset.batch_collate_fn)


    valid_data_paths = [
        "./data/src.valid",
        "./data/dst.valid"
    ]
    ValidDataset = KoKodataset(valid_data_paths)
    ValidDataset.src_vocab = TrainDataset.src_vocab
    ValidDataset.dst_vocab = TrainDataset.dst_vocab

    ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=ValidDataset.batch_collate_fn)

    configs = dict()
    configs['input_dim'] = len(TrainDataset.src_vocab)
    configs['output_dim'] = len(TrainDataset.dst_vocab)
    configs['src_pad_idx'] = 0
    configs['trg_pad_idx'] = 0
    configs['device'] = 'cuda:1'
    configs['epochs'] = 150

    trainer = Trainer(configs)
    trainer.run(TrainDataloader, ValidDataloader)
    return trainer, TrainDataset, ValidDataset, configs

if __name__ == '__main__':
    option = sys.argv[1]

    save_path = "./model"
    if option == "train":
        trainer, TrainDataset, ValidDataset, config = train()
        save_vocab(os.path.join(save_path, 'vocab/train_vocab.pkl'), copy.deepcopy(TrainDataset))

    elif option == "test":
        model_path = os.path.join(save_path, '3_best_model_36.pt') #159
        configs = dict()
        configs['input_dim'] = 3004
        configs['output_dim'] = 3004
        configs['src_pad_idx'] = 0
        configs['trg_pad_idx'] = 0
        configs['device'] = 'cuda:1'
        configs['epochs'] = None
        trainer = Trainer(configs)

        vocabs = read_vocab(os.path.join(save_path, 'vocab/train_vocab.pkl'))
        trainer.model.load_state_dict(torch.load(model_path))
        print("---Load Finish")
        trainer.model.eval()
        while True:
            try:
                sentence = input()
                src_indexes = vocabs.src_vocab.stoi(sentence, option='seq2seq')

                vocabs.dst_vocab.build_index_dict()
                trg_index_dict = vocabs.dst_vocab.index_dict
                
                trg_sos_index = 2
                trg_eos_index = 3
                translation_idx, attention = translate_sentence(src_indexes, trg_sos_index, trg_eos_index, trainer.model, configs['device'], max_len = 50)
                translation = " ".join([trg_index_dict[i] for i in translation_idx])
                print(f'predicted trg = {translation}')
            except Exception as e:
                print(e)