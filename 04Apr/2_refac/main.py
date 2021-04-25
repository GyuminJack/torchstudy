from src.Trainer import *
from src.data import *
from infer import *

def main():
    
    BATCH_SIZE = 64

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

    trainer = Trainer(configs)
    trainer.run(TrainDataloader, ValidDataloader)
    return trainer, TrainDataset, ValidDataset, configs

if __name__ == '__main__':
    trainer, TrainDataset, ValidDataset, config = main()
    while True:
        try:
            
            sentence = input()
            src_indexes = TrainDataset.src_vocab.stoi(sentence, option='seq2seq')
            TrainDataset.dst_vocab.build_index_dict()
            trg_index_dict = TrainDataset.dst_vocab.index_dict
            
            trg_sos_index = 2
            trg_eos_index = 3
            translation_idx, attention = translate_sentence(src_indexes, trg_sos_index, trg_eos_index, trainer.model, config['device'], max_len = 50)
            print(translation_idx)
            translation = " ".join([trg_index_dict[i] for i in translation_idx])
            print(f'predicted trg = {translation}')
        except:
            break