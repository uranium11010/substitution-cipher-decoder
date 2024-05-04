# Substitution cipher decoder

Final project for MIT 6.7800 (6.437).

## Algorithm

1. Use Metropolis-Hastings (MH) to sample from the posterior distribution of the cipher conditioned on the ciphertext.
2. Finetune by increasing the number of words that are in the list of the top 10000 most common words.
In each stage, several runs are conducted in parallel and the best one is kept.

## Instructions

To run all test cases, run
```
python test.py
```
There are 89 test cases; each has 40 to 400 words.

The decoder function (located at [src/decode.py](`src/decode.py`)) is
```
decode(ciphertext: str, has_breakpoint: bool, debug_file_name: str = "test", debug: bool = False) -> str
```
where
* `ciphertext` is the input ciphertext
* `has_breakpoint` is whether there's a breakpoint in the encoding function
(location in the ciphertext where the cipher changes)
* `debug_file_name` is used in the name of the debugging log when `debug` is `True`
* `debug` specifies whether to print debug messages into the debugging log
