import os
from src.encode import clean_text

src_base_path = "data/raw_texts/"
dest_base_path = "data/texts/for_training"
os.makedirs(dest_base_path, exist_ok=True)

to_skip = [
    "cantrbry/cp.html",
    "cantrbry/fields.c",
    "cantrbry/grammar.lsp",
    "cantrbry/kennedy.xls",
    "cantrbry/ptt5",
    "cantrbry/sum",
    "cantrbry/xargs.1",
    "large/E.coli",
    "calgary/bib",
    "calgary/geo",
    "calgary/obj1",
    "calgary/pic",
    "calgary/progl",
    "calgary/trans",
    "calgary/obj2",
    "calgary/progc",
    "calgary/progp",
    "glowbe/sampleSources.xlsx",
]

for corpus_name in os.listdir(src_base_path):
    print(f"CLEANING CORPUS {corpus_name}")
    corpus_path = os.path.join(src_base_path, corpus_name)
    os.makedirs(os.path.join(dest_base_path, corpus_name), exist_ok=True)
    for file_name in os.listdir(corpus_path):
        if os.path.splitext(file_name)[1] == ".zip":
            continue
        if os.path.join(corpus_name, file_name) in to_skip:
            continue
        src_path = os.path.join(corpus_path, file_name)
        print(f"RAW FILE: {src_path}")
        with open(src_path, 'r') as f:
            cleaned_text = clean_text(f.read())
        dest_path = os.path.join(dest_base_path, corpus_name, file_name)
        with open(dest_path, 'w') as f:
            f.write(cleaned_text)
        print(f"CLEANED FILE: {dest_path}")
