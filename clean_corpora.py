import os
import re
from src.encode import clean_text

raw_base_path = "data/raw_texts/gigaword_eng_5/data/"
clean_base_path = "data/training_corpora/"
os.makedirs(clean_base_path, exist_ok=True)

for corpus_name in os.listdir(raw_base_path):
    print(f"CLEANING CORPUS {corpus_name}")
    corpus_path = os.path.join(raw_base_path, corpus_name)
    os.makedirs(os.path.join(clean_base_path, corpus_name), exist_ok=True)
    for file_name in os.listdir(corpus_path):
        raw_path = os.path.join(corpus_path, file_name)
        print(f"RAW FILE: {raw_path}")
        with open(raw_path, 'r') as f:
            raw_text = f.read()
        # remove <DATELINE> ... </DATELINE>
        text = re.sub("<DATELINE>\n.*\n</DATELINE>\n", '', raw_text)
        # remove SGML tags
        text = re.sub("^<.*>$", '', text, flags=re.MULTILINE)
        cleaned_text = clean_text(text)
        clean_path = os.path.join(clean_base_path, corpus_name, file_name)
        with open(clean_path, 'w') as f:
            f.write(cleaned_text)
        print(f"CLEANED FILE: {clean_path}")
