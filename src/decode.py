from typing import Optional, Tuple
from multiprocessing import Pool
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

import numpy as np

from .cipher_array import CipherArray
from .utils import *


def decode_mh(
        cipher_inds: np.ndarray,
        has_breakpoint: bool,
        num_it: int,
        num_ciphers: int,
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> Tuple[CipherArray, np.ndarray, np.number]:
    np.random.seed(seed)
    cipher = CipherArray.new(has_breakpoint, num_ciphers, len(cipher_inds))
    plain_inds = cipher.decode(cipher_inds)
    log_prob = get_trigram_log_prob(plain_inds)
    for it in range(num_it):
        cipher.transition()
        new_plain_inds = cipher.decode(cipher_inds)
        new_log_prob = get_trigram_log_prob(new_plain_inds)
        which_accept = np.log2(np.random.uniform(size=num_ciphers)) < new_log_prob - log_prob
        cipher.accept(which_accept)
        log_prob = np.where(which_accept, new_log_prob, log_prob)
        if debug and (it + 1) % 100 == 0:
            plain_inds = cipher.decode(cipher_inds)
            for i in range(num_ciphers):
                logger.info(f"ITERATION {it + 1}")
                logger.info(f"ACCEPTANCE LOG PROB: {new_log_prob[i] - log_prob[i]}")
                logger.info(f"LOG PROB PER SYMB: {log_prob[i] / len(cipher_inds)}")
                logger.info(inds_to_text(plain_inds[i]))
    best_cipher_idx = np.argmax(log_prob)
    best_cipher = cipher[best_cipher_idx:best_cipher_idx+1]
    best_plain_inds = best_cipher.decode(cipher_inds).reshape(-1)
    best_log_prob = log_prob[best_cipher_idx]
    return best_cipher, best_plain_inds, best_log_prob


def finetune_words(
        plaintext: str,
        bp_word_char_idxs: Optional[Tuple[int, int, int]],
        num_it: int,
        seed: Optional[int] = None,
    ) -> Tuple[str, int]:
    plain_words = plaintext.split()
    np.random.seed(seed)
    num_bad = init_num_bad = get_num_bad_words(map(strip_period, plain_words))
    if num_bad == 0:
        return plaintext, 0
    for it in range(num_it):
        plain_words_swapped, letter1, letter2, left_idx, right_idx = swap_random_letters(plain_words, bp_word_char_idxs)
        new_num_bad = get_num_bad_words(map(strip_period, plain_words_swapped))
        if new_num_bad < num_bad:
            plain_words = plain_words_swapped
            plaintext = swap_letters(plaintext, letter1, letter2, left_idx, right_idx)
            num_bad = new_num_bad
            if num_bad == 0:
                break
    return plaintext, init_num_bad - num_bad


def decode(ciphertext: str, has_breakpoint: bool, test_name: str = "test", debug: bool = False) -> str:
    if debug:
        logging.basicConfig(filename=f"debug_{test_name}.log", level=logging.INFO)
        logging.info('>' * 50 + str(datetime.now()))
    # Hyperparameters
    num_it_mh = 5000
    num_attempts_mh = 4
    num_ciphers_mh = 16
    num_it_ft = 2000
    num_attempts_ft = 4
    # Run several attempts in parallel
    with Pool() as p:
        # M-H with trigram model
        cipher_inds = text_to_inds(ciphertext)
        results_mh = p.starmap(decode_mh,
                               [(cipher_inds, has_breakpoint, num_it_mh, num_ciphers_mh, seed, debug)
                                for seed in range(num_attempts_mh)])
        cipher, plain_inds, log_prob = max(results_mh, key=lambda item: item[2])
        plaintext = inds_to_text(plain_inds)
        if debug:
            logger.info(f"INTERMEDIATE (log prob per symb: {log_prob / len(cipher_inds) :.4f}):")
            logger.info(plaintext)
        # Finetune by increasing number of valid words
        bp_word_char_idxs = cipher.get_bp_word_char_idxs(plaintext)
        results_ft = p.starmap(finetune_words,
                               [(plaintext, bp_word_char_idxs, num_it_ft, seed)
                                for seed in range(num_attempts_ft)])
    plaintext, improvement = max(results_ft, key=lambda item: item[1])
    if debug:
        logger.info(f"FINAL (ft improvement: {improvement}):")
        logger.info(plaintext)
        logger.info('<' * 50 + str(datetime.now()))
    return plaintext
