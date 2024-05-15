import string
import numpy as np

ALPHABET = list(string.ascii_lowercase) + [" ", "."]
LETTER_TO_IDX = dict(map(reversed, enumerate(ALPHABET)))
ALPHABET_SIZE = len(ALPHABET)

P = np.load("begin_2gram_probs_google.npy")
LOG_P = np.log2(P)
LOG_P[np.isinf(LOG_P)] = -25
M = np.load("transitions_3gram_google.npy")
LOG_M = np.log2(M)
LOG_M[np.isinf(LOG_M)] = -25

with open("data/google-10000-english.txt", 'r') as f:
    WORD_LIST = {line[:-1] for line in f.readlines()}
