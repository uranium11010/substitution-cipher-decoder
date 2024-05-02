import numpy as np

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


def swap_letters(word, letter1, letter2, left_idx=0, right_idx=float('inf')):
    return ''.join(letter1 if left_idx <= i < right_idx and letter == letter2
                   else (letter2 if left_idx <= i < right_idx and letter == letter1
                         else letter)
                   for i, letter in enumerate(word))
