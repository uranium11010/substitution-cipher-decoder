import os
import random
random.seed(0)

from src.encode import encode, encode_with_breakpoint

plaintext_files = ["data/texts/feynman.txt", "data/texts/milton.txt", "data/texts/tolstoy.txt"]
test_case_dir = "data/test_cases"

os.mkdir(test_case_dir)

for short in [False, True]:
    for plaintext_file in plaintext_files:
        _, filename_ext = os.path.split(plaintext_file)
        filename, _ = os.path.splitext(filename_ext)
        with open(plaintext_file, 'r') as f:
            plaintext = f.read()
        words = plaintext.split()
        i = 1
        while words:
            test_filename = filename + str(i) + ('s' if short else '')
            print(f"Creating {test_filename}")
            length = random.randint(4, 39) if short else random.randint(40, 400)
            cur_words, words = words[:length], words[length:]
            cur_plaintext = ' '.join(cur_words)
            cur_ciphertext = encode(cur_plaintext)
            cur_ciphertext_bp, _ = encode_with_breakpoint(cur_plaintext)
            with open(os.path.join(test_case_dir, test_filename + ".out"), 'w') as f:
                f.write(cur_plaintext)
            with open(os.path.join(test_case_dir, test_filename + ".in"), 'w') as f:
                f.write(cur_ciphertext)
            with open(os.path.join(test_case_dir, test_filename + "_bp.in"), 'w') as f:
                f.write(cur_ciphertext_bp)
            i += 1
