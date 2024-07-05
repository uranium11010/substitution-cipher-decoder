import os

corpora_dir = "data/training_corpora/"
all_words_path = os.path.join(corpora_dir, "all_words.txt")
with open(all_words_path, 'w') as f_all_words:
    for corpus_name in os.listdir(corpora_dir):
        corpus_path = os.path.join(corpora_dir, corpus_name)
        if not os.path.isdir(corpus_path):
            continue
        corpus_words_path = os.path.join(corpora_dir, f"{corpus_name}_words.txt")
        with open(corpus_words_path, 'w') as f_corpus_words:
            for file_name in sorted(os.listdir(corpus_path), reverse=True):
                file_path = os.path.join(corpus_path, file_name)
                print("ADDING:", file_path)
                with open(file_path, 'r') as f_src:
                    while char := f_src.read(1):
                        char_to_append = '\n' if char == ' ' else char
                        f_corpus_words.write(char_to_append)
                        f_all_words.write(char_to_append)
                f_corpus_words.write('\n')
                f_all_words.write('\n')
