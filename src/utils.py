from typing import Optional, Iterable, Tuple, List

import numpy as np

from .constants import ALPHABET, LETTER_TO_IDX, LOG_P, LOG_M, WORD_LIST


def text_to_inds(text: str) -> np.ndarray:
    return np.array([LETTER_TO_IDX[c] for c in text], dtype=np.int8)


def inds_to_text(inds: np.ndarray) -> str:
    return "".join([ALPHABET[i] for i in inds])


def random_idx_pair(num_idxs: int, skip: Optional[int] = None) -> Tuple[int, int]:
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


def swap_letters(word: str, letter1: str, letter2: str, left_idx: int = 0, right_idx: Optional[int] = None) -> str:
    if right_idx is None:
        right_idx = len(word)
    return ''.join(letter1 if left_idx <= i < right_idx and letter == letter2
                   else (letter2 if left_idx <= i < right_idx and letter == letter1
                         else letter)
                   for i, letter in enumerate(word))


def swap_random_letters(
        words: List[str],
        bp_word_char_idxs: Optional[Tuple[int, int, int]]
    ) -> Tuple[List[str], str, str, int, Optional[int]]:
    i, j = random_idx_pair(len(ALPHABET), skip=LETTER_TO_IDX[' '])
    letter1 = ALPHABET[i]
    letter2 = ALPHABET[j]
    if bp_word_char_idxs is None:
        words_swapped = [swap_letters(word, letter1, letter2) for word in words]
        left_idx = 0
        right_idx = None
    else:
        bp, bp_word_idx, bp_char_idx = bp_word_char_idxs
        left = np.random.uniform() < bp_word_idx / len(words)
        words_swapped = [swap_letters(word, letter1, letter2)
                         if left == (i < bp_word_idx)
                         else word
                         for i, word in enumerate(words)]
        bp_word = words[bp_word_idx]
        bp_word_swapped = swap_letters(bp_word, letter1, letter2,
                                       left_idx = 0 if left else bp_char_idx,
                                       right_idx = bp_char_idx if left else len(bp_word))
        words_swapped[bp_word_idx] = bp_word_swapped
        left_idx = 0 if left else bp
        right_idx = bp if left else None
    return words_swapped, letter1, letter2, left_idx, right_idx


def get_trigram_log_prob(plain_inds: np.ndarray) -> np.ndarray:
    return (LOG_P[plain_inds[...,0], plain_inds[...,1]]
            + np.sum(LOG_M[plain_inds[...,:-2], plain_inds[...,1:-1], plain_inds[...,2:]], axis=-1))


def strip_period(word: str) -> str:
    return word if not word or word[-1] != '.' else word[:-1]


def get_num_bad_words(plain_words: Iterable[str]) -> int:
    return sum(word not in WORD_LIST for word in plain_words)
