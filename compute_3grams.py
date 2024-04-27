import numpy as np

from src.encode import ALPHABET, LETTER_TO_IDX

files = ["data/texts/feynman.txt", "data/texts/milton.txt", "data/texts/tolstoy.txt"]
freqs = np.zeros((len(ALPHABET), len(ALPHABET), len(ALPHABET)), dtype=int)
begin_freqs = np.zeros((len(ALPHABET), len(ALPHABET)), dtype=int)
for file in files:
    with open(file, 'r') as f:
        text = f.read()
    letter_idxs = [LETTER_TO_IDX[c] for c in text]
    begin_freqs[letter_idxs[0], letter_idxs[1]] += 1
    for i in range(len(text) - 2):
        freqs[letter_idxs[i], letter_idxs[i+1], letter_idxs[i+2]] += 1
        if text[i] == ' ' and i + 2 < len(text):
            begin_freqs[letter_idxs[i+1], letter_idxs[i+2]] += 1
transitions = freqs / np.expand_dims(np.sum(freqs, axis=2), 2)
transitions[np.isnan(transitions)] = 1 / len(ALPHABET)
np.save("transitions_3gram.npy", transitions)
begin_freqs = begin_freqs / np.sum(begin_freqs)
np.save("begin_2gram_probs.npy", begin_freqs)
