import logging
logger = logging.getLogger(__name__)

import numpy as np
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


def get_plain_log_prob(cipher_inds, decode_map):
    plain_inds = decode_map[cipher_inds]
    # log_prob = log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    log_prob = log_P[plain_inds[0], plain_inds[1]] + np.sum(log_M[plain_inds[:-2], plain_inds[1:-1], plain_inds[2:]])
    return plain_inds, log_prob


def get_plain_log_prob_bp(cipher_inds, decode_map_l, decode_map_r, bp):
    plain_inds = np.concatenate([decode_map_l[cipher_inds[:bp]], decode_map_r[cipher_inds[bp:]]])
    # log_prob = log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    log_prob = log_P[plain_inds[0], plain_inds[1]] + np.sum(log_M[plain_inds[:-2], plain_inds[1:-1], plain_inds[2:]])
    return plain_inds, log_prob


def decode_once(ciphertext: str, has_breakpoint: bool, N: int, seed=None, test_name="test", debug=False):
    np.random.seed(seed)
    plaintext = ciphertext
    cipher_inds = np.array([LETTER_TO_IDX[c] for c in ciphertext])
    log_probs = []
    acceptance_log_probs = []
    acceptances = []
    win_idx = None
    bp = None
    if has_breakpoint:
        decode_map_l = np.random.permutation(len(ALPHABET))
        decode_map_r = np.random.permutation(len(ALPHABET))
        bp = len(ciphertext) // 2
        plain_inds, log_prob = get_plain_log_prob_bp(cipher_inds, decode_map_l, decode_map_r, bp)
        for it in range(N):
            if np.random.uniform() < 0.9:
                i, j = random_idx_pair(len(ALPHABET))
                if np.random.uniform() < 1/2:
                    new_decode_map_l = np.copy(decode_map_l)
                    new_decode_map_l[i], new_decode_map_l[j] = new_decode_map_l[j], new_decode_map_l[i]
                    new_decode_map_r = decode_map_r
                else:
                    new_decode_map_r = np.copy(decode_map_r)
                    new_decode_map_r[i], new_decode_map_r[j] = new_decode_map_r[j], new_decode_map_r[i]
                    new_decode_map_l = decode_map_l
                new_bp = bp
            else:
                new_bp = bp
                while new_bp == bp or new_bp == 0 or new_bp == len(ciphertext):
                    new_bp = np.random.binomial(len(ciphertext), bp / len(ciphertext))
                new_decode_map_l = decode_map_l
                new_decode_map_r = decode_map_r
            new_plain_inds, new_log_prob = get_plain_log_prob_bp(cipher_inds, new_decode_map_l, new_decode_map_r, new_bp)
            acceptance_log_probs.append(new_log_prob - log_prob)
            if np.random.uniform() < np.exp2(new_log_prob - log_prob):
                decode_map_l = new_decode_map_l
                decode_map_r = new_decode_map_r
                bp = new_bp
                plain_inds = new_plain_inds
                log_prob = new_log_prob
                plaintext = "".join([ALPHABET[i] for i in plain_inds])
                acceptances.append(True)
            else:
                acceptances.append(False)
            log_probs.append(log_prob)
            if debug and (it + 1) % 1000 == 0:
                logger.info(it)
                logger.info(log_prob / len(ciphertext))
                logger.info(plaintext)
    else:
        decode_map = np.random.permutation(len(ALPHABET))
        plain_inds, log_prob = get_plain_log_prob(cipher_inds, decode_map)
        for it in range(N):
            new_decode_map = np.copy(decode_map)
            i, j = random_idx_pair(len(ALPHABET))
            new_decode_map[i], new_decode_map[j] = new_decode_map[j], new_decode_map[i]
            new_plain_inds, new_log_prob = get_plain_log_prob(cipher_inds, new_decode_map)
            acceptance_log_probs.append(new_log_prob - log_prob)
            if np.random.uniform() < np.exp2(new_log_prob - log_prob):
                decode_map = new_decode_map
                plain_inds = new_plain_inds
                log_prob = new_log_prob
                plaintext = "".join([ALPHABET[i] for i in plain_inds])
                acceptances.append(True)
            else:
                acceptances.append(False)
            log_probs.append(log_prob)
            if debug and (it + 1) % 1000 == 0:
                logger.info(it)
                logger.info(log_prob / len(ciphertext))
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
        W = 200
        sliding_sums = [np.sum(acceptances[:W])]
        for t in range(W, len(acceptances)):
            sliding_sums.append(sliding_sums[-1] + acceptances[t] - acceptances[t-W])
        plt.plot(np.arange(W - 1, len(acceptances)), np.array(sliding_sums) / W)
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance rate")
        plt.savefig(f"{test_name}_s{seed}_acceptance.png")
        plt.clf()
        # PLOT DECODING ACCURACY
        # plt.plot(np.arange(N), accs)
        # plt.xlabel("Iteration")
        # plt.ylabel("Decoding accuracy")
        # plt.savefig("acc.png")
    return plaintext, bp, log_probs[-1]


def swap_letters(word, letter1, letter2, left_idx=0, right_idx=float('inf')):
    return ''.join(letter1 if left_idx <= i < right_idx and letter == letter2
                   else (letter2 if left_idx <= i < right_idx and letter == letter1
                         else letter)
                   for i, letter in enumerate(word))


def strip_period(word):
    return word if not word or word[-1] != '.' else word[:-1]


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


def finetune_words(plaintext, plain_words, bp_idxs, N_finetune):
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
    else:
        N = 16000
        num_attempts = 200
        N_finetune = 2000
    with Pool() as p:
        results = p.starmap(decode_once, [(ciphertext, has_breakpoint, N, seed, test_name, debug) for seed in range(num_attempts)])
    plaintext, bp, log_prob = max(results, key=lambda item: item[2])
    if has_breakpoint:
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
    plaintext, improvement = finetune_words(plaintext, plain_words, bp_idxs, N_finetune)
    if debug:
        logger.info(f"Finetune improvement: {improvement}")
        logger.info(f"Final: {plaintext}")
    return plaintext
