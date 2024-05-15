# SPLIT BY TEXT
plaintext_files_dict = {
    "train": ["data/texts/tolstoy.txt"],
    "valid": ["data/texts/feynman.txt"],
    "test": ["data/texts/milton.txt"],
}
for split in ["train", "valid", "test"]:
    plaintext_files = plaintext_files_dict[split]
    data_file = f"data/{split}.txt"
    num_lines = 0
    with open(data_file, 'w') as f_dest:
        for plaintext_file in plaintext_files:
            with open(plaintext_file, 'r') as f_src:
                plaintext = f_src.read()
            length = 20
            for i in range(len(plaintext) // length):
                f_dest.write(plaintext[length * i : length * (i + 1)] + '\n')
            num_lines += len(plaintext) // length
    print(f"{split} set size: {num_lines}")
