from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

from .constants import ALPHABET
from .utils import random_idx_pair


class CipherArray(ABC):
    @staticmethod
    def new(has_bp: bool, num_ciphers: int, ciphertext_len: int) -> 'CipherArray':
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

    @abstractmethod
    def get_bp_word_char_idxs(self, plaintext: str) -> Optional[Tuple[int, int, int]]:
        pass

    @abstractmethod
    def __getitem__(self, index) -> 'CipherArray':
        pass


class CipherNoBpArray(CipherArray):
    def __init__(self, num_ciphers: int, ciphertext_len: int, decode_map=None):
        self.num_ciphers = num_ciphers
        self.ciphertext_len = ciphertext_len

        if decode_map is None:
            self.decode_map = np.vstack([np.random.permutation(len(ALPHABET)) for _ in range(num_ciphers)]).astype(np.int8)
        else:
            self.decode_map = decode_map

        self.last_changes = None

    def transition(self):
        self.last_changes = [random_idx_pair(len(ALPHABET)) for _ in range(self.num_ciphers)]
        for k, (i, j) in enumerate(self.last_changes):
            self.decode_map[k,i], self.decode_map[k,j] = self.decode_map[k,j], self.decode_map[k,i]

    def accept(self, acceptances):
        for k, (acceptance, (i, j)) in enumerate(zip(acceptances, self.last_changes)):
            if not acceptance:
                self.decode_map[k,i], self.decode_map[k,j] = self.decode_map[k,j], self.decode_map[k,i]

    def decode(self, cipher_inds: np.ndarray) -> np.ndarray:
        return self.decode_map[:, cipher_inds]

    def get_bp_word_char_idxs(self, plaintext: str) -> Optional[Tuple[int, int, int]]:
        return None

    def __getitem__(self, index) -> 'CipherNoBpArray':
        assert isinstance(index, slice)
        new_decode_map = self.decode_map[index]
        return CipherNoBpArray(len(new_decode_map), self.ciphertext_len, decode_map=new_decode_map)


class CipherBpArray(CipherArray):
    def __init__(self, num_ciphers: int, ciphertext_len: int, decode_map_l=None, decode_map_r=None, bp=None):
        self.num_ciphers = num_ciphers
        self.ciphertext_len = ciphertext_len

        if decode_map_l is None:
            self.decode_map_l = np.vstack([np.random.permutation(len(ALPHABET)) for _ in range(num_ciphers)]).astype(np.int8)
        else:
            self.decode_map_l = decode_map_l
        if decode_map_r is None:
            self.decode_map_r = np.vstack([np.random.permutation(len(ALPHABET)) for _ in range(num_ciphers)]).astype(np.int8)
        else:
            self.decode_map_r = decode_map_r
        if bp is None:
            self.bp = (ciphertext_len // 2) * np.ones(num_ciphers, dtype=np.int32)
        else:
            self.bp = bp

        self.last_changes = None

    def transition(self):
        self.last_changes = []
        for k in range(self.num_ciphers):
            if np.random.uniform() < 0.9:
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
                while new_bp == self.bp[k]:
                    clipped_bp = max(1, min(self.bp[k], self.ciphertext_len - 1))
                    new_bp = np.random.binomial(self.ciphertext_len, clipped_bp / self.ciphertext_len)
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
        left_mask = np.arange(len(cipher_inds), dtype=np.int32)[None, :] < self.bp[:, None]
        plain_inds = np.where(left_mask, self.decode_map_l[:, cipher_inds], self.decode_map_r[:, cipher_inds])
        return plain_inds

    def get_bp_word_char_idxs(self, plaintext: str) -> Optional[Tuple[int, int, int]]:
        bp = self.bp[0]
        plaintext_bp = plaintext[:bp] + '|' + plaintext[bp:]
        plain_words_bp = plaintext_bp.split()
        for bp_word_idx in range(len(plain_words_bp)):
            bp_char_idx = plain_words_bp[bp_word_idx].find('|')
            if bp_char_idx != -1:
                break
        bp_word = plain_words_bp[bp_word_idx]
        bp_word = bp_word[:bp_char_idx] + bp_word[bp_char_idx+1:]
        return bp, bp_word_idx, bp_char_idx

    def __getitem__(self, index) -> 'CipherBpArray':
        assert isinstance(index, slice)
        new_decode_map_l = self.decode_map_l[index]
        new_decode_map_r = self.decode_map_r[index]
        new_bp = self.bp[index]
        return CipherBpArray(len(new_bp), self.ciphertext_len,
                decode_map_l=new_decode_map_l, decode_map_r=new_decode_map_r, bp=new_bp)
