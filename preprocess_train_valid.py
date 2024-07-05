import os
import random

random.seed(0)

lengths = [256, 128, 64, 32, 16]
corpus = "nyt_eng"
words_path = f"data/training_corpora/{corpus}_words.txt"
max_dataset_size = 10000000
dataset_sizes = {length: 0 for length in lengths}
for target_length in lengths:
    print("TARGET LENGTH:", target_length)
    train_valid_path = f"data/train_valid_{corpus}_{target_length}.txt"
    num_so_far = 0
    running_words = []
    with open(words_path, 'r') as f_words:
        with open(train_valid_path, 'w') as f_train_valid:
            running_length = -1
            while True:
                if running_length < target_length:
                    try:
                        next_word = next(f_words)[:-1]
                    except StopIteration:
                        break
                    running_words.append(next_word)
                    running_length += len(next_word) + 1
                else:
                    if running_length == target_length:
                        f_train_valid.write(' '.join(running_words) + '\n')
                        dataset_sizes[target_length] += 1
                        if dataset_sizes[target_length] % 1000000 == 0:
                            print(dataset_sizes[target_length])
                        if dataset_sizes[target_length] == max_dataset_size:
                            break
                    forgotten_word = running_words.pop(0)
                    running_length -= len(forgotten_word) + 1
print("DATASET SIZES:")
print(dataset_sizes)
