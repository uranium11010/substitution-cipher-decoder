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

from .constants import ALPHABET, LETTER_TO_IDX
from .utils import text_to_inds, random_idx_pair, swap_letters, NumpyArraySet

# P = np.array(pd.read_csv("data/letter_probabilities.csv", header=None))[0]
P = np.load("begin_2gram_probs_google.npy")
log_P = np.log2(P)
log_P[np.isinf(log_P)] = -25
# M = np.array(pd.read_csv("data/letter_transition_matrix.csv", header=None))
M = np.load("transitions_3gram_google.npy")
log_M = np.log2(M)
log_M[np.isinf(log_M)] = -25

with open("data/google-10000-english.txt", 'r') as f:
    # word_list = NumpyArraySet(text_to_inds(line[:-1]) for line in f.readlines())
    word_list = {line[:-1] for line in f.readlines()}
# with open("data/google-10000-english-oneaway.txt", 'r') as f:
#     # word_oneaway_list = NumpyArraySet(text_to_inds(line[:-1]) for line in f.readlines())
#     word_oneaway_list = {line[:-1] for line in f.readlines()}
# with open("data/google-books-common-words.txt", 'r') as f:
#     word_list = {line[:line.find('\t')].lower() for line in f.readlines()}
# word_list_dict = defaultdict(list)
# for word_inds in word_list:
#     word_list_dict[len(word_inds)].append(word_inds)
# word_array_dict = {}
# for word_length, words in word_list_dict.items():
#     word_array_dict[word_length] = np.vstack(words)


def get_log_prob(plain_inds: np.ndarray, use_words=False):
    # return log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    trigram_log_prob = log_P[plain_inds[...,0], plain_inds[...,1]] + np.sum(log_M[plain_inds[...,:-2], plain_inds[...,1:-1], plain_inds[...,2:]], axis=-1)
    if not use_words:
        return trigram_log_prob
    # space_locs = np.arange(len(plain_inds), dtype=int)[plain_inds == LETTER_TO_IDX[' ']]
    # space_locs = np.concatenate([np.array([-1], dtype=int), space_locs, np.array([len(plain_inds)])])
    total_mismatch_count = 0
    num_words = 0
    left = 0
    done = False
    while left < len(plain_inds):
        while plain_inds[left] == LETTER_TO_IDX[' '] or plain_inds[left] == LETTER_TO_IDX['.']:
            left += 1
            if left >= len(plain_inds):
                done = True
                break
        if done:
            break
        right = left + 1
        while right < len(plain_inds) and plain_inds[right] != LETTER_TO_IDX[' '] and plain_inds[right] != LETTER_TO_IDX['.']:
            right += 1
        total_mismatch_count += (right - left) * (plain_inds[left:right] not in word_list)
        num_words += 1
        left = right + 1
    # plain_word_list_dict = {word_length: [] for word_length in word_array_dict}
    # plain_word_lengths = space_locs[1:] - space_locs[:-1] - 1
    # for i, word_length in enumerate(plain_word_lengths):
    #     if word_length in word_array_dict:
    #         plain_word_list_dict[word_length].append(plain_inds[space_locs[i]+1:space_locs[i+1]])
    #     else:
    #         total_mismatch_count += word_length
    # for word_length, plain_word_list in plain_word_list_dict.items():
    #     if not plain_word_list:
    #         continue
    #     plain_word_array = np.vstack(plain_word_list)
    #     mismatch_counts = np.sum(np.expand_dims(plain_word_array, axis=1) != np.expand_dims(word_array_dict[word_length], axis=0), axis=2)
    #     best_mismatches = np.min(mismatch_counts, axis=1)
    #     total_mismatch_count += np.sum(best_mismatches)
    expected_num_words = len(plain_inds) / 5.7
    num_words_log_prob = -expected_num_words * np.log2(np.e) + num_words * np.log2(expected_num_words) - np.sum(np.log2(np.arange(1, num_words + 1)))
    return -total_mismatch_count + num_words_log_prob + trigram_log_prob


class CipherArray(ABC):
    @staticmethod
    def new(has_bp: bool, num_ciphers: int, ciphertext_len: int):
        if has_bp:
            return CipherBpArray(num_ciphers, ciphertext_len)
        return CipherNoBpArray(num_ciphers, ciphertext_len)

    @abstractmethod
    def transition(self, finetune=False) -> None:
        pass

    @abstractmethod
    def accept(self, acceptances) -> None:
        pass

    @abstractmethod
    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        pass


class CipherNoBpArray(CipherArray):
    def __init__(self, num_ciphers: int, ciphertext_len: int, decode_map=None):
        self.num_ciphers = num_ciphers
        self.ciphertext_len = ciphertext_len

        if decode_map is None:
            self.decode_map = np.vstack([np.random.permutation(len(ALPHABET)) for _ in range(num_ciphers)], dtype=np.int8)
        else:
            self.decode_map = decode_map

        self.last_changes = None

    def transition(self, finetune=False):
        self.last_changes = [random_idx_pair(len(ALPHABET)) for _ in range(self.num_ciphers)]
        for k, (i, j) in enumerate(self.last_changes):
            self.decode_map[k,i], self.decode_map[k,j] = self.decode_map[k,j], self.decode_map[k,i]

    def accept(self, acceptances):
        for k, (acceptance, (i, j)) in enumerate(zip(acceptances, self.last_changes)):
            if not acceptance:
                self.decode_map[k,i], self.decode_map[k,j] = self.decode_map[k,j], self.decode_map[k,i]

    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        return self.decode_map[:, cipher_inds]

    def __getitem__(self, index) -> 'CipherNoBpArray':
        assert isinstance(index, slice)
        new_decode_map = self.decode_map[index]
        return CipherNoBpArray(len(new_decode_map), self.ciphertext_len, decode_map=new_decode_map)


class CipherBpArray(CipherArray):
    def __init__(self, num_ciphers: int, ciphertext_len: int, decode_map_l=None, decode_map_r=None, bp=None):
        self.num_ciphers = num_ciphers
        self.ciphertext_len = ciphertext_len

        if decode_map_l is None:
            self.decode_map_l = np.vstack([np.random.permutation(len(ALPHABET)) for _ in range(num_ciphers)], dtype=np.int8)
        else:
            self.decode_map_l = decode_map_l
        if decode_map_r is None:
            self.decode_map_r = np.vstack([np.random.permutation(len(ALPHABET)) for _ in range(num_ciphers)], dtype=np.int8)
        else:
            self.decode_map_r = decode_map_r
        if bp is None:
            self.bp = (ciphertext_len // 2) * np.ones(num_ciphers, dtype=np.int32)
        else:
            self.bp = bp

        self.last_changes = None

    def transition(self, finetune=False) -> 'CipherBpArray':
        self.last_changes = []
        for k in range(self.num_ciphers):
            if np.random.uniform() < (1.0 if finetune else 0.9):
                i, j = random_idx_pair(len(ALPHABET))
                if np.random.uniform() < 0.5:
                    self.decode_map_l[k,i], self.decode_map_l[k,j] = self.decode_map_l[k,j], self.decode_map_l[k,i]
                    self.last_changes.append(("left", (i, j)))
                else:
                    self.decode_map_r[k,i], self.decode_map_r[k,j] = self.decode_map_r[k,j], self.decode_map_r[k,i]
                    self.last_changes.append(("right", (i, j)))
            else:
                self.last_changes.append(("bp", self.bp[k]))
                new_bp = self.bp[k]
                while new_bp == self.bp[k] or new_bp == 0 or new_bp == self.ciphertext_len:
                    new_bp = np.random.binomial(self.ciphertext_len, self.bp[k] / self.ciphertext_len)
                self.bp[k] = new_bp

    def accept(self, acceptances):
        for k, (acceptance, (change_type, change)) in enumerate(zip(acceptances, self.last_changes)):
            if not acceptance:
                if change_type != "bp":
                    i, j = change
                    if change_type == "left":
                        self.decode_map_l[k,i], self.decode_map_l[k,j] = self.decode_map_l[k,j], self.decode_map_l[k,i]
                    elif change_type == "right":
                        self.decode_map_r[k,i], self.decode_map_r[k,j] = self.decode_map_r[k,j], self.decode_map_r[k,i]
                    else:
                        raise Exception("Non-existent change type")
                else:
                    self.bp[k] = change

    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        # plain_inds = np.vstack([
        #     np.concatenate([decode_map_l_row[cipher_inds[:bp]], decode_map_r_row[cipher_inds[bp:]]])
        #     for decode_map_l_row, decode_map_r_row, bp in zip(self.decode_map_l, self.decode_map_r, self.bp)
        # ])
        left_mask = np.arange(len(cipher_inds), dtype=np.int32)[None, :] < self.bp[:, None]
        plain_inds = np.where(left_mask, self.decode_map_l[:, cipher_inds], self.decode_map_r[:, cipher_inds])
        return plain_inds

    def __getitem__(self, index) -> 'CipherBpArray':
        assert isinstance(index, slice)
        new_decode_map_l = self.decode_map_l[index]
        new_decode_map_r = self.decode_map_r[index]
        new_bp = self.bp[index]
        return CipherBpArray(len(new_bp), self.ciphertext_len,
                decode_map_l=new_decode_map_l, decode_map_r=new_decode_map_r, bp=new_bp)


def decode_once(ciphertext: str, has_breakpoint: bool, N: int, init_cipher=None, finetune=False, seed=None, test_name="test", debug=False):
    if debug:
        log_probs = []
        acceptance_log_probs = []
        acceptances = []
    np.random.seed(seed)
    cipher_inds = text_to_inds(ciphertext)
    num_ciphers = 16
    cur_best_cipher = cipher = CipherArray.new(has_breakpoint, num_ciphers, len(ciphertext)) if init_cipher is None else init_cipher
    cur_best_plain_inds = plain_inds = cipher.decode(cipher_inds)
    cur_best_log_prob = log_prob = get_log_prob(plain_inds, use_words=finetune)
    for it in range(N):
        cipher.transition(finetune=finetune)
        new_plain_inds = cipher.decode(cipher_inds)
        new_log_prob = get_log_prob(new_plain_inds, use_words=finetune)
        if debug:
            acceptance_log_probs.append(new_log_prob - log_prob)
        which_accept = np.log2(np.random.uniform(size=num_ciphers)) < new_log_prob - log_prob
        cipher.accept(which_accept)
        log_prob = np.where(which_accept, new_log_prob, log_prob)
        # if np.random.uniform() < np.exp2(new_log_prob - log_prob):
        #     cipher = new_cipher
        #     plain_inds = new_plain_inds
        #     log_prob = new_log_prob
        #     if log_prob > cur_best_log_prob:
        #         cur_best_log_prob = log_prob
        #         cur_best_plain_inds = plain_inds
        #         cur_best_cipher = cipher
        #     if debug:
        #         acceptances.append(True)
        # elif debug:
        #     acceptances.append(False)
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
    # return cur_best_cipher, cur_best_plain_inds, cur_best_log_prob
    best_cipher_idx = np.argmax(log_prob)
    best_cipher = cipher[best_cipher_idx:best_cipher_idx+1]
    best_plain_inds = best_cipher.decode(cipher_inds).reshape(-1)
    best_log_prob = log_prob[best_cipher_idx]
    return best_cipher, best_plain_inds, best_log_prob


def strip_period(word):
    return word if not word or word[-1] != '.' else word[:-1]


def get_num_bad(plain_words):
    num_bad = 0
    for word in plain_words:
        stripped_word = strip_period(word)
        if stripped_word not in word_list:
            # if stripped_word not in word_oneaway_list:
            #     num_bad += len(stripped_word)
            # else:
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
        num_attempts_finetune = 8
    else:
        N = 16000
        num_attempts = 200
        N_finetune = 2000
        num_attempts_finetune = 8
    with Pool() as p:
        results = p.starmap(decode_once, [(ciphertext, has_breakpoint, N, None, False, seed, test_name, False) for seed in range(num_attempts)])
        cipher, plain_inds, log_prob = max(results, key=lambda item: item[2])
        plaintext = "".join([ALPHABET[i] for i in plain_inds])
        logger.info(f"INTERMEDIATE: {plaintext}")
        if has_breakpoint:
            bp = cipher.bp[0]
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
        # logger.info(f"Finetune improvement: {improvement}")
        logger.info(f"FINAL: {plaintext}")
    return plaintext
