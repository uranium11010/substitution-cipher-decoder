from src.encode import clean_text

text_paths = [
    ("data/cantrbry/alice29.txt", "data/texts/alice29.txt"),
    ("data/cantrbry/asyoulik.txt", "data/texts/asyoulik.txt"),
    ("data/cantrbry/plrabn12.txt", "data/texts/plrabn12.txt"),
    ("data/cantrbry/lcet10.txt", "data/texts/lcet10.txt"),
    ("data/large/bible.txt", "data/texts/bible.txt"),
    ("data/large/world192.txt", "data/texts/world192.txt"),
    ("data/calgary/book1", "data/texts/book1.txt"),
    ("data/calgary/book2", "data/texts/book2.txt"),
    ("data/calgary/paper1", "data/texts/paper1.txt"),
    ("data/calgary/paper2", "data/texts/paper2.txt"),
]
for src_path, dest_path in text_paths:
    print(f"CLEANING {src_path}")
    with open(src_path, 'r') as f:
        cleaned_text = clean_text(f.read())
    with open(dest_path, 'w') as f:
        f.write(cleaned_text)
    print(f"WROTE TO {dest_path}")
