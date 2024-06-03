import os
import random

random.seed(0)

length = 20
# TRAIN AND VALID SET
train_valid_src_dir = "data/texts/for_training/"
train_valid_list = []
for corpus_name in os.listdir(train_valid_src_dir):
    corpus_path = os.path.join(train_valid_src_dir, corpus_name)
    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, 'r') as f_src:
            while True:
                plaintext = f_src.read(length)
                if len(plaintext) < length:
                    break
                train_valid_list.append(plaintext)
random.shuffle(train_valid_list)
num_data = len(train_valid_list)
valid_set_size = num_data // 9
train_set_size = num_data - valid_set_size
train_dest_file = "data/train.txt"
with open(train_dest_file, 'w') as f_dest:
    for i in range(train_set_size):
        f_dest.write(train_valid_list[i] + '\n')
print(f"train set size: {train_set_size}")
valid_dest_file = "data/valid.txt"
with open(valid_dest_file, 'w') as f_dest:
    for i in range(train_set_size, num_data):
        f_dest.write(train_valid_list[i] + '\n')
print(f"valid set size: {valid_set_size}")
# TEST SET
test_src_files = ["data/texts/tolstoy.txt", "data/texts/feynman.txt", "data/texts/milton.txt"]
test_dest_file = "data/test.txt"
num_lines = 0
with open(test_dest_file, 'w') as f_dest:
    for test_src_file in test_src_files:
        with open(test_src_file, 'r') as f_src:
            while True:
                plaintext = f_src.read(length)
                if len(plaintext) < length:
                    break
                f_dest.write(plaintext + '\n')
                num_lines += 1
print(f"test set size: {num_lines}")
