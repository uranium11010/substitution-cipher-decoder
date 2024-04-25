import numpy as np
import pandas as pd
import matplotlib
from multiprocessing import Pool
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

from .encode import ALPHABET, LETTER_TO_IDX

P = np.array(pd.read_csv("data/letter_probabilities.csv", header=None))[0]
log_P = np.log2(P)
M = np.array(pd.read_csv("data/letter_transition_matrix.csv", header=None))
log_M = np.log2(M)
log_M[np.isinf(log_M)] = -1e9

with open("data/google-10000-english.txt", 'r') as f:
    word_list = {line[:-1] for line in f.readlines()}

with open("data/sample/plaintext.txt", 'r') as f:
    gold_plaintext = f.read()
# print(gold_plaintext)
# print(len(gold_plaintext))


def get_plain_log_prob(cipher_inds, decode_map):
    plain_inds = decode_map[cipher_inds]
    log_prob = log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    return plain_inds, log_prob


def get_plain_log_prob_bp(cipher_inds, decode_map_l, decode_map_r, bp):
    plain_inds = np.concatenate([decode_map_l[cipher_inds[:bp]], decode_map_r[cipher_inds[bp:]]])
    log_prob = log_P[plain_inds[0]] + np.sum(log_M[plain_inds[1:], plain_inds[:-1]])
    return plain_inds, log_prob


def decode_once(ciphertext: str, has_breakpoint: bool, N: int, seed=None, debug=False):
    gold_plain_inds = np.array([LETTER_TO_IDX[c] for c in gold_plaintext[:len(ciphertext)]])
    np.random.seed(seed)
    plaintext = ciphertext
    cipher_inds = np.array([LETTER_TO_IDX[c] for c in ciphertext])
    log_probs = []
    acceptances = []
    accs = []
    win_idx = None
    if has_breakpoint:
        decode_map_l = np.random.permutation(len(ALPHABET))
        decode_map_r = np.random.permutation(len(ALPHABET))
        bp = len(ciphertext) // 2
        plain_inds, log_prob = get_plain_log_prob_bp(cipher_inds, decode_map_l, decode_map_r, bp)
        for it in range(N):
            if np.random.uniform() < 0.9:
                i = np.random.choice(len(ALPHABET))
                j = np.random.choice(len(ALPHABET) - 1)
                if j >= i:
                    j += 1
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
            a = np.exp2(new_log_prob - log_prob)
            # print(a)
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
            accs.append(np.mean(plain_inds == gold_plain_inds))
            if win_idx is None and accs[-1] >= 0.99:
                win_idx = it
            if debug and (it + 1) % 1000 == 0:
                print(it)
                print(accs[-1])
                print(plaintext)
        # print("win:", win_idx)
    else:
        decode_map = np.random.permutation(len(ALPHABET))
        plain_inds, log_prob = get_plain_log_prob(cipher_inds, decode_map)
        for it in range(N):
            new_decode_map = np.copy(decode_map)
            i = np.random.choice(len(ALPHABET))
            j = np.random.choice(len(ALPHABET) - 1)
            if j >= i:
                j += 1
            new_decode_map[i], new_decode_map[j] = new_decode_map[j], new_decode_map[i]
            new_plain_inds, new_log_prob = get_plain_log_prob(cipher_inds, new_decode_map)
            a = np.exp2(new_log_prob - log_prob)
            # print(a)
            if np.random.uniform() < np.exp2(new_log_prob - log_prob):
                decode_map = new_decode_map
                plain_inds = new_plain_inds
                log_prob = new_log_prob
                plaintext = "".join([ALPHABET[i] for i in plain_inds])
                acceptances.append(True)
            else:
                acceptances.append(False)
            log_probs.append(log_prob)
            accs.append(np.mean(plain_inds == gold_plain_inds))
            if debug and (it + 1) % 1000 == 0:
                print(it)
                print(accs[-1])
                print(plaintext)
    # print(''.join(np.array(['0', '1'])[(plain_inds == gold_plain_inds).astype(int)]))
    # PLOT LOG PROB OF ACCEPTED STATE
    # plt.subplots_adjust(left=0.2)
    # plt.plot(np.arange(N), np.array(log_probs) / len(ciphertext))
    # plt.ylim([-11, 0])
    # plt.xlabel("Iteration")
    # plt.ylabel("Log probability per symbol (bits)")
    # plt.savefig("log_probs_per_sym.png")
    # print(log_probs[-1] / len(ciphertext))
    # PLOT ACCEPTANCE RATIO
    # W = 200
    # sliding_sums = [np.sum(acceptances[:W])]
    # for t in range(W, len(acceptances)):
    #     sliding_sums.append(sliding_sums[-1] + acceptances[t] - acceptances[t-W])
    # plt.plot(np.arange(W - 1, len(acceptances)), np.array(sliding_sums) / W)
    # plt.xlim([0, 2600])
    # plt.xlabel("Iteration")
    # plt.ylabel("Acceptance rate")
    # plt.savefig("acceptance.png")
    # PLOT DECODING ACCURACY
    # plt.plot(np.arange(N), accs)
    # plt.xlabel("Iteration")
    # plt.ylabel("Decoding accuracy")
    # plt.savefig("acc.png")
    return plaintext, log_probs[-1]


def swap_letters(word, letter1, letter2):
    return ''.join(letter1 if letter == letter2 else (letter2 if letter == letter1 else letter) for letter in word)


def decode(ciphertext: str, has_breakpoint: bool, debug=False) -> str:
    if has_breakpoint:
        N = 20000
        num_attempts = 160
    else:
        N = 16000
        num_attempts = 200
    with Pool() as p:
        results = p.starmap(decode_once, [(ciphertext, has_breakpoint, N, seed, debug) for seed in range(num_attempts)])
    plaintext, log_prob = max(results, key=lambda item: item[1])
    plain_words = [word if word[-1] != '.' else word[:-1] for word in plaintext.split()]
    plain_bad_words = [word for word in plain_words if word not in word_list]
    if any(swap_letters(word, 'j', 'q') in word_list for word in plain_bad_words):
        plaintext = swap_letters(plaintext, 'j', 'q')
    return plaintext
