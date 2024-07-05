import os
import random

random.seed(0)

test_src_file = "data/test_texts/history_wiki.txt"
with open(test_src_file, 'r') as f_src:
    src_text = f_src.read()
lengths = [256, 128, 64, 32, 16]
cases_per_length = 50
for length in lengths:
    allowed_starts = []
    start = 0
    while start <= len(src_text) - length:
        end = start + length
        if end == len(src_text) or src_text[end] == ' ':
            allowed_starts.append(start)
            start = end + 1
        else:
            while start < len(src_text) and src_text[start] != ' ':
                start += 1
            start += 1
    starts = random.sample(allowed_starts, cases_per_length)
    with open(f"data/test_{length}.txt", 'w') as f_dest:
        for start in starts:
            end = start + length
            f_dest.write(src_text[start:end] + '\n')
