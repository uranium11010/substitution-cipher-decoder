from __future__ import annotations

from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)


import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib
from multiprocessing import Pool
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .encode import ALPHABET, LETTER_TO_IDX

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


def get_log_prob(plain_inds: NDArray):
    # return log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    return log_P[plain_inds[0], plain_inds[1]] + np.sum(log_M[plain_inds[:-2], plain_inds[1:-1], plain_inds[2:]])


def random_idx_pair(num_idxs, skip=None):
    i = np.random.choice(num_idxs - (skip is not None))
    j = np.random.choice(num_idxs - 1 - (skip is not None))
    if j >= i:
        j += 1
    if skip is not None:
        if i >= skip:
            i += 1
        if j >= skip:
            j += 1
    return i, j


class Cipher(ABC):
    @staticmethod
    def new(has_bp: bool, ciphertext_len: int):
        if has_bp:
            return CipherBp(ciphertext_len)
        return CipherNoBp(ciphertext_len)

    @abstractmethod
    def transition(self) -> Cipher:
        pass

    @abstractmethod
    def decode(self, cipher_inds: NDArray) -> NDArray:
        pass


class CipherNoBp(Cipher):
    def __init__(self, ciphertext_len, decode_map=None):
        self.ciphertext_len = ciphertext_len
        if decode_map is None:
            self.decode_map = np.random.permutation(len(ALPHABET))
        else:
            self.decode_map = decode_map

    def transition(self) -> CipherNoBp:
        new_decode_map = np.copy(self.decode_map)
        i, j = random_idx_pair(len(ALPHABET))
        new_decode_map[i], new_decode_map[j] = new_decode_map[j], new_decode_map[i]
        return CipherNoBp(self.ciphertext_len, new_decode_map)

    def decode(self, cipher_inds: NDArray) -> NDArray:
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

    def transition(self) -> CipherBp:
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

    def decode(self, cipher_inds: NDArray) -> NDArray:
        plain_inds = np.concatenate([self.decode_map_l[cipher_inds[:self.bp]], self.decode_map_r[cipher_inds[self.bp:]]])
        return plain_inds


def decode_once(ciphertext: str, has_breakpoint: bool, N: int, seed=None, test_name="test", debug=False):
    if debug:
        log_probs = []
        acceptance_log_probs = []
        acceptances = []
    np.random.seed(seed)
    cipher_inds = np.array([LETTER_TO_IDX[c] for c in ciphertext])
    cipher = Cipher.new(has_breakpoint, len(ciphertext))
    plain_inds = cipher.decode(cipher_inds)
    log_prob = get_log_prob(plain_inds)
    for it in range(N):
        new_cipher = cipher.transition()
        new_plain_inds = new_cipher.decode(cipher_inds)
        new_log_prob = get_log_prob(new_plain_inds)
        if debug:
            acceptance_log_probs.append(new_log_prob - log_prob)
        if np.random.uniform() < np.exp2(new_log_prob - log_prob):
            cipher = new_cipher
            plain_inds = new_plain_inds
            log_prob = new_log_prob
            if debug:
                acceptances.append(True)
        elif debug:
            acceptances.append(False)
        if debug:
            log_probs.append(log_prob)
            if (it + 1) % 1 == 0:
                logger.info(it)
                logger.info(f"ACCEPTANCE LOG PROB: {acceptance_log_probs[-1]}")
                logger.info(f"LOG PROB PER SYMB: {log_prob / len(ciphertext)}")
                plaintext = "".join([ALPHABET[i] for i in plain_inds])
                logger.info(plaintext)
    if debug:
        # PLOT LOG PROB OF ACCEPTANCE RATIO
        acceptance_probs = np.exp2(acceptance_log_probs)
        acceptance_probs[acceptance_probs > 2.2] = 2.2
        plt.scatter(np.arange(N), acceptance_probs, s=1, marker='.')
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance ratio")
        plt.savefig(f"{test_name}_s{seed}_probs_A.png")
        plt.clf()
        # PLOT LOG PROB OF ACCEPTED STATE
        plt.subplots_adjust(left=0.2)
        plt.plot(np.arange(N), np.array(log_probs) / len(ciphertext))
        plt.ylim([-11, 0])
        plt.xlabel("Iteration")
        plt.ylabel("Log probability per symbol (bits)")
        plt.savefig(f"{test_name}_s{seed}_log_probs_per_sym.png")
        plt.clf()
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
    return cipher, plain_inds, log_prob


def swap_letters(word, letter1, letter2, left_idx=0, right_idx=float('inf')):
    return ''.join(letter1 if left_idx <= i < right_idx and letter == letter2
                   else (letter2 if left_idx <= i < right_idx and letter == letter1
                         else letter)
                   for i, letter in enumerate(word))


def strip_period(word):
    return word if not word or word[-1] != '.' else word[:-1]


def finetune_words(plaintext, plain_words, bp_idxs, N_finetune, seed=None):
    np.random.seed(seed)
    num_bad = init_num_bad = sum(strip_period(word) not in word_list for word in plain_words)
    if num_bad == 0:
        return plaintext, 0
    if bp_idxs is None:
        for it in range(N_finetune):
            i, j = random_idx_pair(len(ALPHABET), skip=LETTER_TO_IDX[' '])
            letter1 = ALPHABET[i]
            letter2 = ALPHABET[j]
            plain_words_swapped = [swap_letters(word, letter1, letter2) for word in plain_words]
            new_num_bad = sum(strip_period(word_swapped) not in word_list for word_swapped in plain_words_swapped)
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
            new_num_bad = sum(strip_period(word_swapped) not in word_list for word_swapped in plain_words_swapped)
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
        results = p.starmap(decode_once, [(ciphertext, has_breakpoint, N, seed, test_name, debug) for seed in range(num_attempts)])
        cipher, plain_inds, log_prob = max(results, key=lambda item: item[2])
        plaintext = "".join([ALPHABET[i] for i in plain_inds])
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
    plaintext, improvement = max(finetune_results, key=lambda item: item[1])
    if debug:
        logger.info(f"Finetune improvement: {improvement}")
        logger.info(f"Final: {plaintext}")
    return plaintext
