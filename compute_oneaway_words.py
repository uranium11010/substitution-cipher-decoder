from string import ascii_lowercase
from src.utils import random_idx_pair, swap_letters

input_path = "data/google-10000-english.txt"
output_path = "data/google-10000-english-oneaway.txt"
with open(input_path, 'r') as f:
    word_list = {line[:-1] for line in f.readlines()}
oneaway_word_list = set()
for word in word_list:
    for i in range(len(ascii_lowercase) - 1):
        for j in range(i + 1, len(ascii_lowercase)):
            oneaway_word = swap_letters(word, ascii_lowercase[i], ascii_lowercase[j])
            if oneaway_word not in word_list:
                oneaway_word_list.add(oneaway_word)
with open(output_path, 'w') as f:
    for oneaway_word in oneaway_word_list:
        f.write(oneaway_word + '\n')
