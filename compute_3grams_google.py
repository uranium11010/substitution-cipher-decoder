import numpy as np
import pandas as pd

from src.encode import ALPHABET, LETTER_TO_IDX

letter_probs = np.array(pd.read_csv("data/letter_probabilities.csv", header=None))[0]
period_prob = letter_probs[LETTER_TO_IDX['.']] / letter_probs[LETTER_TO_IDX[' ']]
space_prob = 1 - period_prob
transition_probs = np.array(pd.read_csv("data/letter_transition_matrix.csv", header=None))
bigram_probs = transition_probs * letter_probs
rev_transition_probs = bigram_probs / np.expand_dims(np.sum(bigram_probs, axis=1), axis=1)
transition_probs = transition_probs.T

file = "data/google-books-common-words.txt"
freqs = np.zeros((len(ALPHABET), len(ALPHABET), len(ALPHABET)))
begin_freqs = np.zeros((len(ALPHABET), len(ALPHABET)))
f = open(file, 'r')
while line := f.readline():
    word, count = line.split('\t')
    word = word.lower()
    count = int(count)
    letter_idxs = [LETTER_TO_IDX[c] for c in word]
    if len(word) > 1:
        begin_freqs[letter_idxs[0], letter_idxs[1]] += count
        freqs[LETTER_TO_IDX[' '], letter_idxs[0], letter_idxs[1]] += count
    else:
        begin_freqs[letter_idxs[0], LETTER_TO_IDX['.']] += count * period_prob
        begin_freqs[letter_idxs[0], LETTER_TO_IDX[' ']] += count * space_prob
        freqs[LETTER_TO_IDX[' '], letter_idxs[0], LETTER_TO_IDX['.']] += count * period_prob
        freqs[LETTER_TO_IDX[' '], letter_idxs[0], LETTER_TO_IDX[' ']] += count * space_prob
    freqs[LETTER_TO_IDX['.'], LETTER_TO_IDX[' '], letter_idxs[0]] += count * period_prob
    for i in range(len(word) - 2):
        freqs[letter_idxs[i], letter_idxs[i+1], letter_idxs[i+2]] += count
    if len(word) > 1:
        freqs[letter_idxs[-2], letter_idxs[-1], LETTER_TO_IDX[' ']] += count * space_prob
        freqs[letter_idxs[-2], letter_idxs[-1], LETTER_TO_IDX['.']] += count * period_prob
    freqs[letter_idxs[-1], LETTER_TO_IDX['.'], LETTER_TO_IDX[' ']] += count * period_prob
    freqs[letter_idxs[-1], LETTER_TO_IDX[' ']] += count * space_prob * transition_probs[LETTER_TO_IDX[' ']]
f.close()
marginals = np.sum(freqs, axis=2)
trigram_transitions = freqs / np.expand_dims(marginals, 2)
trigram_transitions[marginals == 0] = letter_probs
np.save("transitions_3gram_google.npy", trigram_transitions)
begin_freqs = begin_freqs / np.sum(begin_freqs)
np.save("begin_2gram_probs_google.npy", begin_freqs)
