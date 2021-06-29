# https://keep-steady.tistory.com/37#recentEntries

import os
from konlpy.tag import Mecab
import torch
import numpy as np
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizerFast


def apply_mecab_files(text_files, save_path, generation=False):
    ret_save_paths = []
    mecab_tokenizer = Mecab().morphs
    total_morph = []

    for text_path in text_files:
        # load korean corpus for tokenizer training
        with open(text_path, "r", encoding="utf-8") as f:
            data = f.read().split("\n")

        for sentence in data:
            morph_sentence = []
            count = 0
            for token_mecab in mecab_tokenizer(sentence):
                token_mecab_save = token_mecab
                if count > 0:
                    token_mecab_save = "##" + token_mecab_save  # 앞에 ##를 부친다
                    morph_sentence.append(token_mecab_save)
                else:
                    morph_sentence.append(token_mecab_save)
                    count += 1
            total_morph.append(morph_sentence)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for line in total_morph:
                f.write(" ".join(line) + "\n")

    return save_path


def select_tokenizer(tn_type="BWT"):
    # 4가지중 tokenizer 선택
    how_to_tokenize = BertWordPieceTokenizer  # The famous Bert tokenizer, using WordPiece
    # how_to_tokenize = SentencePieceBPETokenizer  # A BPE implementation compatible with the one used by SentencePiece
    # how_to_tokenize = CharBPETokenizer  # The original BPE
    # how_to_tokenize = ByteLevelBPETokenizer  # The byte level version of the BPE

    # Initialize a tokenizer
    if str(how_to_tokenize) == str(BertWordPieceTokenizer):
        ## 주의!! 한국어는 strip_accents를 False로 해줘야 한다
        # 만약 True일 시 나는 -> 'ㄴ','ㅏ','ㄴ','ㅡ','ㄴ' 로 쪼개져서 처리된다
        # 학습시 False했으므로 load할 때도 False를 꼭 확인해야 한다
        tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
    elif str(how_to_tokenize) == str(SentencePieceBPETokenizer):
        tokenizer = SentencePieceBPETokenizer()

    elif str(how_to_tokenize) == str(CharBPETokenizer):
        tokenizer = CharBPETokenizer()

    elif str(how_to_tokenize) == str(ByteLevelBPETokenizer):
        tokenizer = ByteLevelBPETokenizer()
    else:
        assert "select right tokenizer"
    return tokenizer


def train_tokenizer(tokenizer, corpus_files, vocab_size, limit_alphabet, save_path=None, save_name="Default", min_freq=5):

    tokenizer.train(
        files=corpus_files,
        vocab_size=vocab_size,
        min_frequency=min_freq,  # 단어의 최소 발생 빈도, 5
        limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
        show_progress=True,
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_model(save_path)

    return tokenizer


def load_tokenizer(tokenizer_path):
    loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, strip_accents=False, lowercase=False)  # Must be False if cased model  # 로드
    return loaded_tokenizer


if __name__ == "__main__":
    import sys
    import glob
    import datetime
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--corpus_path", default="/home/jack/torchstudy/06Jun/BERT/data/wpm/*.txt")
    parser.add_argument("--save_name", default="peti_namu")
    parser.add_argument("--vocab_size", default=12000)
    parser.add_argument("--load_path", default="/home/jack/torchstudy/06Jun/BERT/vocabs/default_2021062318/")
    parser.add_argument("--min_freq", default=3)
    args = parser.parse_args()

    if args.mode == "train":
        now = datetime.datetime.now()
        files = glob.glob(args.corpus_path)
        print(f"Wordpiece train target : {files}")

        # -- Apply Sentence to Mecab tokenizer
        print("Apply Mecab to Raw Sentences")
        # files = ["/home/jack/torchstudy/05May/ELMo/data/sentence_cleaned_namu_train.ko"]
        corpus_save_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/.tmpraw/splited_by_mecab_{}.txt".format(now.strftime("%Y%m%d%H"))
        corpus_path = apply_mecab_files(files, corpus_save_path)

        print("Mecab Finish (temp corpus file saved : {})".format(os.path.abspath(corpus_save_path)))
        # -- Make Vocab.txt file
        tokenizer = select_tokenizer()

        vocab_save_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/{}_{}".format(args.save_name, now.strftime("%Y%m%d%H"))
        
        train_tokenizer(
            tokenizer=tokenizer,
            corpus_files=[corpus_path],
            vocab_size=args.vocab_size,
            limit_alphabet=6000,
            save_path=vocab_save_path,
            min_freq=args.min_freq,
        )

    elif args.mode == "test":
        vocab_save_path = args.load_path
        tokenizer = load_tokenizer(vocab_save_path)
        sentence = ["나는 오늘 아침밥을 먹었다.", "우리집 강아지는 복슬강아지 학교갔다"]

        output = tokenizer.encode(sentence, return_tensors="pt")

        print("No Mecab Tokens (str) : {}".format([tokenizer.convert_ids_to_tokens(s) for s in output.tolist()]))

        mecab_tokenizer = Mecab().morphs
        sentence = [" ".join(mecab_tokenizer(i)) for i in sentence]

        output = tokenizer.encode(sentence, return_tensors="pt")

        print("Yes Mecab Tokens (str) : {}".format([tokenizer.convert_ids_to_tokens(s) for s in output.tolist()]))
