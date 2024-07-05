"""
Script for generating ciphertexts.
Usage: python3 encode.py plaintext.out ciphertext.out has_breakpoint [seed]

Behavior:
    1. Reads in standard input (until EOF).
    2. Cleans text to satisfy requirements given in the project handout.
    3. Writes the cleaned text to `plaintext.out`.
    4. Encodes the cleaned text and writes the ciphertext to `ciphertext.out`.

Setting has_breakpoint to true encodes with a breakpoint.
Passing a seed as the optional last argument makes the encoding deterministic.

Example invocations:
    python3 encode.py plaintext.txt ciphertext.txt false 42 < data/texts/feynman.txt
    python3 encode.py plaintext.txt ciphertext.txt true < data/texts/tolstoy.txt

Can also be used for just cleaning text in the following way:
    python3 encode.py clean.txt /dev/null 0 < dirty.txt
"""

import sys
import string
import re
import random
import unicodedata

from .constants import ALPHABET, LETTER_TO_IDX


def _clean_text(text: str) -> str:
    # try and approximate unicode with ascii
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    # make lowercase
    text = text.lower()
    # replace whitespace characters with space
    text = re.sub(r"\s", ' ', text)
    # remove invalid characters
    text = ''.join(filter(ALPHABET.__contains__, text))
    # strip spaces, replace contiguous spaces with one
    text = ' '.join(text.split())
    assert len(text) > 0, "Cannot have empty text!"
    return text


def assert_clean(text: str):
    assert _clean_text(text) == text

    assert len(text) > 0
    assert all(x in ALPHABET for x in text)
    assert text[0] in string.ascii_lowercase
    for i, x in enumerate(text):
        if x == " ":
            assert text[i + 1] in string.ascii_lowercase


def clean_text(text: str) -> str:
    clean = _clean_text(text)
    assert_clean(clean)
    return clean


def encode(plaintext: str) -> str:
    cipherbet = ALPHABET.copy()
    random.shuffle(cipherbet)

    ciphertext = "".join(cipherbet[LETTER_TO_IDX[c]] for c in plaintext)
    return ciphertext


def main():
    plaintext_out = sys.argv[1]
    ciphertext_out = sys.argv[2]
    has_breakpoint = (sys.argv[3].lower() == "true")
    if len(sys.argv) > 4:
        random.seed(sys.argv[4])

    raw_text = sys.stdin.read()

    plaintext = clean_text(raw_text)
    print(f"Clean plaintext length: {len(plaintext)}")
    with open(plaintext_out, "w") as f:
        f.write(plaintext)

    if has_breakpoint:
        print("Encoding with breakpoint...")
        ciphertext, bpoint = encode_with_breakpoint(plaintext)
        print(f"Breakpoint at position {bpoint}")
    else:
        print("Encoding without breakpoint")
        ciphertext = encode(plaintext)

    with open(ciphertext_out, "w") as f:
        f.write(ciphertext)


if __name__ == "__main__":
    main()
