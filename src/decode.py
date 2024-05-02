from abc import ABC, abstractmethod
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


import numpy as np
import pandas as pd
import matplotlib
from multiprocessing import Pool
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .encode import ALPHABET, LETTER_TO_IDX
from .utils import random_idx_pair, swap_letters

# P = np.array(pd.read_csv("data/letter_probabilities.csv", header=None))[0]
P = np.load("begin_2gram_probs_google.npy")
log_P = np.log2(P)
log_P[np.isinf(log_P)] = -25
# M = np.array(pd.read_csv("data/letter_transition_matrix.csv", header=None))
M = np.load("transitions_3gram_google.npy")
log_M = np.log2(M)
log_M[np.isinf(log_M)] = -25

with open("data/google-10000-english.txt", 'r') as f:
    word_list = {line[:-1] for line in f.readlines()}
with open("data/google-10000-english-oneaway.txt", 'r') as f:
    word_oneaway_list = {line[:-1] for line in f.readlines()}
# with open("data/google-books-common-words.txt", 'r') as f:
#     word_list = {line[:line.find('\t')].lower() for line in f.readlines()}
word_list_dict = defaultdict(list)
for word in word_list:
    word_inds = np.array([LETTER_TO_IDX[c] for c in word])
    word_list_dict[len(word)].append(word_inds)
word_array_dict = {}
for word_length, words in word_list_dict.items():
    word_array_dict[word_length] = np.vstack(words)


def get_log_prob(plain_inds: np.ndarray, use_words=False):
    # return log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    trigram_log_prob = log_P[plain_inds[0], plain_inds[1]] + np.sum(log_M[plain_inds[:-2], plain_inds[1:-1], plain_inds[2:]])
    if not use_words:
        return trigram_log_prob
    space_locs = np.arange(len(plain_inds), dtype=int)[plain_inds == LETTER_TO_IDX[' ']]
    space_locs = np.concatenate([np.array([-1], dtype=int), space_locs, np.array([len(plain_inds)])])
    plain_word_list_dict = {word_length: [] for word_length in word_array_dict}
    plain_word_lengths = space_locs[1:] - space_locs[:-1] - 1
    total_mismatch_count = 0
    for i, word_length in enumerate(plain_word_lengths):
        if word_length in word_array_dict:
            plain_word_list_dict[word_length].append(plain_inds[space_locs[i]+1:space_locs[i+1]])
        else:
            total_mismatch_count += word_length
    for word_length, plain_word_list in plain_word_list_dict.items():
        if not plain_word_list:
            continue
        plain_word_array = np.vstack(plain_word_list)
        mismatch_counts = np.sum(np.expand_dims(plain_word_array, axis=1) != np.expand_dims(word_array_dict[word_length], axis=0), axis=2)
        best_mismatches = np.min(mismatch_counts, axis=1)
        total_mismatch_count += np.sum(best_mismatches)
    return -total_mismatch_count * 10 + trigram_log_prob


class Cipher(ABC):
    @staticmethod
    def new(has_bp: bool, ciphertext_len: int):
        if has_bp:
            return CipherBp(ciphertext_len)
        return CipherNoBp(ciphertext_len)

    @abstractmethod
    def transition(self) -> 'Cipher':
        pass

    @abstractmethod
    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        pass


class CipherNoBp(Cipher):
    def __init__(self, ciphertext_len, decode_map=None):
        self.ciphertext_len = ciphertext_len
        if decode_map is None:
            self.decode_map = np.random.permutation(len(ALPHABET))
        else:
            self.decode_map = decode_map

    def transition(self) -> 'CipherNoBp':
        new_decode_map = np.copy(self.decode_map)
        i, j = random_idx_pair(len(ALPHABET))
        new_decode_map[i], new_decode_map[j] = new_decode_map[j], new_decode_map[i]
        return CipherNoBp(self.ciphertext_len, new_decode_map)

    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        return self.decode_map[cipher_inds]


class CipherBp(Cipher):
    def __init__(self, ciphertext_len, decode_map_l=None, decode_map_r=None, bp=None):
        self.ciphertext_len = ciphertext_len
        if decode_map_l is None:
            self.decode_map_l = np.random.permutation(len(ALPHABET))
        else:
            self.decode_map_l = decode_map_l
        if decode_map_r is None:
            self.decode_map_r = np.random.permutation(len(ALPHABET))
        else:
            self.decode_map_r = decode_map_r
        if bp is None:
            self.bp = ciphertext_len // 2
        else:
            self.bp = bp

    def transition(self) -> 'CipherBp':
        if np.random.uniform() < 0.9:
            i, j = random_idx_pair(len(ALPHABET))
            if np.random.uniform() < 0.5:
                new_decode_map_l = np.copy(self.decode_map_l)
                new_decode_map_l[i], new_decode_map_l[j] = new_decode_map_l[j], new_decode_map_l[i]
                new_decode_map_r = self.decode_map_r
            else:
                new_decode_map_r = np.copy(self.decode_map_r)
                new_decode_map_r[i], new_decode_map_r[j] = new_decode_map_r[j], new_decode_map_r[i]
                new_decode_map_l = self.decode_map_l
            new_bp = self.bp
        else:
            new_bp = self.bp
            while new_bp == self.bp or new_bp == 0 or new_bp == self.ciphertext_len:
                new_bp = np.random.binomial(self.ciphertext_len, self.bp / self.ciphertext_len)
            new_decode_map_l = self.decode_map_l
            new_decode_map_r = self.decode_map_r
        return CipherBp(self.ciphertext_len, new_decode_map_l, new_decode_map_r, new_bp)

    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        plain_inds = np.concatenate([self.decode_map_l[cipher_inds[:self.bp]], self.decode_map_r[cipher_inds[self.bp:]]])
        return plain_inds


def decode_once(ciphertext: str, has_breakpoint: bool, N: int, init_cipher=None, finetune=False, seed=None, test_name="test", debug=False):
    if debug:
        log_probs = []
        acceptance_log_probs = []
        acceptances = []
    np.random.seed(seed)
    cipher_inds = np.array([LETTER_TO_IDX[c] for c in ciphertext])
    cur_best_cipher = cipher = Cipher.new(has_breakpoint, len(ciphertext)) if init_cipher is None else init_cipher
    cur_best_plain_inds = plain_inds = cipher.decode(cipher_inds)
    cur_best_log_prob = log_prob = get_log_prob(plain_inds, use_words=finetune)
    for it in range(N):
        new_cipher = cipher.transition()
        new_plain_inds = new_cipher.decode(cipher_inds)
        new_log_prob = get_log_prob(new_plain_inds, use_words=finetune)
        if debug:
            acceptance_log_probs.append(new_log_prob - log_prob)
        if np.random.uniform() < np.exp2(new_log_prob - log_prob):
            cipher = new_cipher
            plain_inds = new_plain_inds
            log_prob = new_log_prob
            if log_prob > cur_best_log_prob:
                cur_best_log_prob = log_prob
                cur_best_plain_inds = plain_inds
                cur_best_cipher = cipher
            if debug:
                acceptances.append(True)
        elif debug:
            acceptances.append(False)
        if debug:
            log_probs.append(log_prob)
            if (it + 1) % 100 == 0:
                logger.info(it)
                logger.info(f"ACCEPTANCE LOG PROB: {acceptance_log_probs[-1]}")
                logger.info(f"LOG PROB PER SYMB: {log_prob / len(ciphertext)}")
                plaintext = "".join([ALPHABET[i] for i in plain_inds])
                logger.info(plaintext)
    if debug:
        pass
        # PLOT LOG PROB OF ACCEPTANCE RATIO
        # acceptance_probs = np.exp2(acceptance_log_probs)
        # acceptance_probs[acceptance_probs > 2.2] = 2.2
        # plt.scatter(np.arange(N), acceptance_probs, s=1, marker='.')
        # plt.xlabel("Iteration")
        # plt.ylabel("Acceptance ratio")
        # plt.savefig(f"{test_name}_s{seed}_probs_A.png")
        # plt.clf()
        # PLOT LOG PROB OF ACCEPTED STATE
        # plt.subplots_adjust(left=0.2)
        # plt.plot(np.arange(N), np.array(log_probs) / len(ciphertext))
        # plt.ylim([-11, 0])
        # plt.xlabel("Iteration")
        # plt.ylabel("Log probability per symbol (bits)")
        # plt.savefig(f"{test_name}_s{seed}_log_probs_per_sym.png")
        # plt.clf()
        # PLOT ACCEPTANCE RATE
        # W = 200
        # sliding_sums = [np.sum(acceptances[:W])]
        # for t in range(W, len(acceptances)):
        #     sliding_sums.append(sliding_sums[-1] + acceptances[t] - acceptances[t-W])
        # plt.plot(np.arange(W - 1, len(acceptances)), np.array(sliding_sums) / W)
        # plt.xlabel("Iteration")
        # plt.ylabel("Acceptance rate")
        # plt.savefig(f"{test_name}_s{seed}_acceptance.png")
        # plt.clf()
        # PLOT DECODING ACCURACY
        # plt.plot(np.arange(N), accs)
        # plt.xlabel("Iteration")
        # plt.ylabel("Decoding accuracy")
        # plt.savefig("acc.png")
    return cur_best_cipher, cur_best_plain_inds, cur_best_log_prob


def strip_period(word):
    return word if not word or word[-1] != '.' else word[:-1]


def get_num_bad(plain_words):
    num_bad = 0
    for word in plain_words:
        stripped_word = strip_period(word)
        if stripped_word not in word_list:
            if stripped_word not in word_oneaway_list:
                num_bad += 2
            else:
                num_bad += 1
    return num_bad


def finetune_words(plaintext, plain_words, bp_idxs, N_finetune, seed=None):
    np.random.seed(seed)
    num_bad = init_num_bad = get_num_bad(plain_words)
    if num_bad == 0:
        return plaintext, 0
    if bp_idxs is None:
        for it in range(N_finetune):
            i, j = random_idx_pair(len(ALPHABET), skip=LETTER_TO_IDX[' '])
            letter1 = ALPHABET[i]
            letter2 = ALPHABET[j]
            plain_words_swapped = [swap_letters(word, letter1, letter2) for word in plain_words]
            new_num_bad = get_num_bad(plain_words_swapped)
            if new_num_bad < num_bad:
                plain_words = plain_words_swapped
                plaintext = swap_letters(plaintext, letter1, letter2)
                num_bad = new_num_bad
                if num_bad == 0:
                    break
    else:
        bp, bp_word_idx, bp_char_idx = bp_idxs
        for it in range(N_finetune):
            i, j = random_idx_pair(len(ALPHABET), skip=LETTER_TO_IDX[' '])
            letter1 = ALPHABET[i]
            letter2 = ALPHABET[j]
            left = np.random.uniform() < bp_word_idx / len(plain_words)
            plain_words_swapped = [swap_letters(word, letter1, letter2)
                                   if left == (i < bp_word_idx)
                                   else word
                                   for i, word in enumerate(plain_words)]
            bp_word = plain_words[bp_word_idx]
            bp_word_swapped = swap_letters(bp_word, letter1, letter2,
                                           left_idx=0 if left else bp_char_idx,
                                           right_idx=bp_char_idx if left else len(bp_word))
            plain_words_swapped[bp_word_idx] = bp_word_swapped
            new_num_bad = get_num_bad(plain_words_swapped)
            if new_num_bad < num_bad:
                plain_words = plain_words_swapped
                plaintext = swap_letters(plaintext, letter1, letter2,
                                         left_idx=0 if left else bp,
                                         right_idx=bp if left else len(plaintext))
                num_bad = new_num_bad
                if num_bad == 0:
                    break
    return plaintext, init_num_bad - num_bad


def decode(ciphertext: str, has_breakpoint: bool, test_name: str = "test", debug: bool = False) -> str:
    if debug:
        logging.basicConfig(filename=f"debug_{test_name}.log", level=logging.INFO)
    np.random.seed(69420)
    if has_breakpoint:
        N = 20000
        num_attempts = 160
        N_finetune = 2000
        num_attempts_finetune = 4
    else:
        N = 16000
        num_attempts = 200
        N_finetune = 2000
        num_attempts_finetune = 4
    with Pool() as p:
        results = p.starmap(decode_once, [(ciphertext, has_breakpoint, N, None, False, seed, test_name, False) for seed in range(num_attempts)])
        cipher, plain_inds, log_prob = max(results, key=lambda item: item[2])
        plaintext = "".join([ALPHABET[i] for i in plain_inds])
        logger.info(f"INTERMEDIATE: {plaintext}")
        if has_breakpoint:
            bp = cipher.bp
            plaintext_bp = plaintext[:bp] + '|' + plaintext[bp:]
            plain_words = plaintext_bp.split()
            for bp_word_idx in range(len(plain_words)):
                bp_char_idx = plain_words[bp_word_idx].find('|')
                if bp_char_idx != -1:
                    break
            bp_word = plain_words[bp_word_idx]
            bp_word = bp_word[:bp_char_idx] + bp_word[bp_char_idx+1:]
            plain_words[bp_word_idx] = bp_word
            bp_idxs = (bp, bp_word_idx, bp_char_idx)
        else:
            plain_words = plaintext.split()
            bp_idxs = None
        finetune_results = p.starmap(finetune_words, [(plaintext, plain_words, bp_idxs, N_finetune, seed) for seed in range(num_attempts_finetune)])
        # finetune_results = p.starmap(decode_once, [(ciphertext, has_breakpoint, N_finetune, cipher, True, seed, test_name, debug) for seed in range(num_attempts_finetune)])
    plaintext, improvement = max(finetune_results, key=lambda item: item[1])
    # cipher, plain_inds, log_prob = max(finetune_results, key=lambda item: item[2])
    # plaintext = "".join([ALPHABET[i] for i in plain_inds])
    if debug:
        logger.info(f"Finetune improvement: {improvement}")
        logger.info(f"FINAL: {plaintext}")
    return plaintext
