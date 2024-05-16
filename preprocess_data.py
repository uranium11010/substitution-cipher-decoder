# SPLIT BY TEXT
plaintext_files_dict = {
    "train": ["data/texts/alice29.txt", "data/texts/book1.txt", "data/texts/book2.txt",
              "data/texts/lcet10.txt", "data/texts/paper1.txt", "data/texts/plrabn12.txt"],
    "valid": ["data/texts/asyoulik.txt", "data/texts/paper2.txt"],
    "test": ["data/texts/tolstoy.txt", "data/texts/feynman.txt", "data/texts/milton.txt"],
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
