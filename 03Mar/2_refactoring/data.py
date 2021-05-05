import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

class EnFrData:
    def __init__(self, config):
        TRG = Field(tokenize = self.tokenize_en, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)

        SRC = Field(tokenize = self.tokenize_fr, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)

        self.spacy_en = spacy.load('en_core_web_sm')
        self.spacy_fr = spacy.load('fr_core_news_sm')
        
        train_data, valid_data, test_data = Multi30k.splits(exts = ('.en', '.fr'), root='/home/jack/torchstudy/03Mar/2_refactoring/.data', fields = (SRC, TRG))

        SRC.build_vocab(train_data, min_freq = 2)
        TRG.build_vocab(train_data, min_freq = 2)
        self.SRC = SRC
        self.TRG = TRG
        BATCH_SIZE = config['batch_size']
        device = config['device']
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = BATCH_SIZE,
                device = device)
    
    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def tokenize_fr(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_fr.tokenizer(text)]
    
    
