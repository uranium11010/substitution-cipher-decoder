# Plaintext-only training of neural substitution cipher decoder

## Task

A *substitution cipher* is a cipher that maps each symbol in the plaintext to a symbol according to a permutation of the alphabet.
Our task is to design a neural architecture that decodes a ciphertext encoded with substitution cipher.

## Architecture

For each letter of the alphabet, encode the positions at which the letter occurs in the input as a multihot vector.
These multihot vectors are passed through a Transformer encoder, outputing a distribution over the alphabet
for each input letter of the alphabet. This defines the mapping from ciphertext letters to plaintext letters.
Since the Transformer architecture is permutation-equivariant, we only need to train on plaintext
and the resultant model will automatically work on ciphertext.

## Results

[TODO]

## Instructions

We use Python >=3.9 with PyTorch and NumPy.

Training and validation data:
* [English Gigaword Fifth Edition](https://catalog.ldc.upenn.edu/LDC2011T07)
After obtaining the data (`gigaword_eng_5_LDC2011T07.tgz`),
create a new directory `data/raw_texts/` and place it there. Extract the contents:
```
cd data/raw_texts
tar -xvzf gigaword_eng_5_LDC2011T07.tgz
gzip -d gigaword_eng_5/data/*/*
cd ../..
```
Then run the following scripts to create the training and validation sets:
```
python clean_corpora.py
python combine_corpora.py
python preprocess_train_valid.py
python split_train_valid.py
```
Run the following scripts to create the test set:
```
python clean_test_text.py
python preprocess_test.py
```

## Statistics of datasets

Stats for entirety of cleaned Gigaword corpora:
* 3958192089 words in total
* Word length frequencies:
```
{3: 0.1826, 4: 0.1551, 2: 0.1495, 5: 0.1155, 6: 0.0984, 7: 0.0952, 8: 0.0657, 9: 0.0477, 10: 0.0287, 1: 0.0286, 11: 0.0151, 12: 0.0082, 13: 0.0049, 14: 0.0022, 15: 0.0009, 16: 0.0005, 17: 0.0003, 18: 0.0003, 19: 0.0002, 20: 0.0001, 21: 0.0001, 22: 0.0001}
```
```
{3: 722651123, 4: 613922646, 2: 591806497, 5: 457046424, 6: 389578802, 7: 376927570, 8: 260158424, 9: 188981773, 10: 113419403, 1: 113278173, 11: 59732540, 12: 32284822, 13: 19327836, 14: 8693910, 15: 3549076, 16: 1916882, 17: 1256912, 18: 1016705, 19: 684238, 20: 421362, 21: 282256, 22: 224261, 23: 163578, 24: 143021, 25: 123863, 26: 98980, 27: 76396, 28: 65743, 29: 55953, 30: 46345, 31: 38514, 32: 31510, 33: 27299, 34: 25278, 35: 18800, 36: 18721, 37: 13909, 38: 11134, 39: 10198, 40: 8775, 41: 7694, 42: 6325, 43: 5490, 44: 4701, 45: 4628, 46: 4162, 47: 3234, 48: 2641, 49: 2334, 50: 2037, 51: 1647, 52: 1449, 53: 1250, 54: 1050, 55: 853, 56: 741, 57: 543, 58: 426, 60: 281, 59: 273, 61: 133, 62: 92, 63: 72, 64: 72, 66: 59, 67: 59, 65: 46, 68: 40, 72: 13, 71: 12, 70: 10, 69: 7, 77: 6, 73: 5, 86: 5, 89: 5, 82: 4, 74: 3, 80: 3, 76: 3, 78: 2, 91: 2, 103: 2, 92: 2, 88: 2, 102: 1, 99: 1, 75: 1, 97: 1, 85: 1, 83: 1, 139: 1, 118: 1, 87: 1, 84: 1, 96: 1, 143: 1, 105: 1, 98: 1, 79: 1, 109: 1, 93: 1, 81: 1, 94: 1}
```
Starts for NYT corpus:
* 1414893849 words in total
* Word length frequencies:
```
{3: 0.1913, 4: 0.1659, 2: 0.1515, 5: 0.1156, 6: 0.0918, 7: 0.0884, 8: 0.0603, 9: 0.0429, 1: 0.0333, 10: 0.0267, 11: 0.0144, 12: 0.0078, 13: 0.0046, 14: 0.0022, 15: 0.001, 16: 0.0005, 17: 0.0004, 18: 0.0003, 19: 0.0003, 20: 0.0002, 21: 0.0001, 22: 0.0001, 23: 0.0001, 24: 0.0001}
```
