from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def bleu_score(real, pred):
    return sentence_bleu(real, pred, weights=(0.25, 0.25, 0.25, 0.25))

def perplexity(probs):
    return np.power(np.prod(probs), -(1 / len(probs)))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

