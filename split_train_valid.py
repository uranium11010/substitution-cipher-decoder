import random

random.seed(0)

lengths = [256, 128, 64, 32, 16]
valid_ratio = 0.1
corpus = "nyt_eng"
for target_length in lengths:
    print("TARGET LENGTH:", target_length)
    train_valid_path = f"data/train_valid_{corpus}_{target_length}.txt"
    dataset_size = 0
    with open(train_valid_path, 'r') as f_train_valid:
        for line in f_train_valid:
            dataset_size += 1
    print("DATASET SIZE:", dataset_size)
    valid_set_size = int(valid_ratio * dataset_size)
    print("VALID SET SIZE:", valid_set_size)
    train_set_size = dataset_size - valid_set_size
    print("TRAIN SET SIZE:", train_set_size)
    total_remaining = dataset_size
    valid_remaining = valid_set_size
    train_path = f"data/train_{corpus}_{target_length}.txt"
    valid_path = f"data/valid_{corpus}_{target_length}.txt"
    with open(train_path, 'w') as f_train:
        with open(valid_path, 'w') as f_valid:
            with open(train_valid_path, 'r') as f_train_valid:
                for i, line in enumerate(f_train_valid):
                    if random.uniform(0, 1) < valid_remaining / total_remaining:
                        f_valid.write(line)
                        valid_remaining -= 1
                    else:
                        f_train.write(line)
                    total_remaining -= 1
                    if (i + 1) % 1000000 == 0:
                        print(i + 1)
