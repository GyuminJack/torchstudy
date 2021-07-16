import os
from transformers import BertTokenizerFast

def klue_to_text_and_bio(path):
    corpus_text = []
    corpus_bio = []
    with open(path, 'r') as f:
        _tokens = []
        _bio = []
        for cnt, line in enumerate(f.readlines()):
            if ("##" not in line) and (line != "\n"):
                line = line.replace("\n", "")
                token, bio = line.split("\t")
                _tokens.append(token)
                _bio.append(bio)
            
            elif line == "\n":
                assert len(_tokens) == len(_bio), "Size Mismatched"
                corpus_text.append("".join(_tokens))
                corpus_bio.append(_bio)
                _tokens = []
                _bio = []
                
    return corpus_text, corpus_bio

def load_tokenizer(tokenizer_path):
    loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, strip_accents=False, lowercase=False)
    return loaded_tokenizer

def get_token_labels(tokenizer, text:str, original_bio:list):
    cleaned_original_bio = [lbl for txt, lbl in list(zip(text, original_bio)) if txt.strip()]
    
    tokenized = tokenizer(text, return_offsets_mapping = True)
    token_list = tokenized['input_ids'][1:-1]
    offset_list = tokenized['offset_mapping'][1:-1]
    
    start_index = 0
    merged_bio = []
    for offset in offset_list:
        token_length = offset[1] - offset[0]
        seleceted_labels = cleaned_original_bio[start_index : start_index+token_length][0] # 가장 첫번째 bio 태그를 태그로 사용
        merged_bio.append(seleceted_labels)
        start_index += token_length
    
    assert len(token_list) == len(merged_bio), "Size Mismatched"
    if len(token_list) != len(merged_bio):
        print("aDfasdklj;fjas")
    return token_list, merged_bio

def save_ner_data(save_path, tokenizer, text:list, bio:list):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for _text, _bio in zip(text, bio):
            _, new_bio = get_token_labels(tokenizer, _text, _bio)
            f.write(_text + "\t" + ",".join(new_bio) + "\n")

if __name__ == "__main__":
    import datetime
    corpus_text, corpus_bio = klue_to_text_and_bio("../data/klue-ner-v1/klue-ner-v1_dev.tsv")
    tokenizer_path =  "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    tokenizer_name = tokenizer_path.split("/")[-1]
    tokenizer = load_tokenizer(tokenizer_path)

    save_path = "../data/{}/klue_ner_{}.dev".format(tokenizer_name, datetime.datetime.now().strftime("%Y%m%d"))
    save_ner_data(save_path, tokenizer, corpus_text, corpus_bio)