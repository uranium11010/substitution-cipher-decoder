import os
from src.encode import clean_text

raw_base_path = "data/raw_texts/"
clean_base_path = "data/test_texts/"
os.makedirs(clean_base_path, exist_ok=True)

for file_name in os.listdir(raw_base_path):
    raw_path = os.path.join(raw_base_path, file_name)
    print(f"RAW FILE: {raw_path}")
    with open(raw_path, 'r') as f:
        raw_text = f.read()
    cleaned_text = clean_text(raw_text)
    clean_path = os.path.join(clean_base_path, file_name)
    with open(clean_path, 'w') as f:
        f.write(cleaned_text)
    print(f"CLEANED FILE: {clean_path}")
