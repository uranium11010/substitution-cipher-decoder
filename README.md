# Substitution cipher decoder

Final project for MIT 6.7800 (6.437). My writeup is [here](writeup.pdf).

## Task

A *substitution cipher* is a cipher that maps each symbol in the plaintext to a symbol according to a permutation of the alphabet.
A *breakpoint* is a location in the ciphertext where the cipher changes.
Our task is to design an algorithm that decodes a ciphertext that is encoded with a single substitution cipher ("no-breakpoint setting")
or two substitution ciphers on two sides of a breakpoint ("breakpoint setting").

## Algorithm

1. Use Metropolis-Hastings (MH) to sample from the posterior distribution of the cipher conditioned on the ciphertext.
We use a trigram model of English to model the plaintext, as computed using [`compute_3grams_google.py`](compute_3grams_google.py).
The proposal distribution in the no-breakpoint setting is a uniformly random transposition (swap of two symbols)
of the permutation defining the cipher.
In the breakpoint setting:
    * With probability $p_{\text{bp}} = 0.1$, change the location $b$ of the breakpoint to $b'$ sampled from the binomial distribution
    $\mathcal{B}(n, p)$, where $n$ is the length of the text and $p = \mathrm{clip}(b/n, 1, n-1)$.
    (We resample if $b' = b$.)
    * With probability $1 - p_{\text{bp}} = 0.9$, choose one of the two substitution ciphers uniformly at random and apply
    a uniformly random transposition to the permutation defining the cipher.

    We run $A_\text{MH} = 512$ attempts of MH each for $N_\text{MH} = 5000$ iterations.
    We keep the result of the best run in terms of the log probability of the decoded plaintext under the trigram model.
    This result enters the finetuning stage below.
2. Finetune by increasing the number of words that are in the list of the top 10000 most common words.
We randomly choose two non-space symbols of the cipher to swap. (In the breakpoint setting, we choose the cipher to the left of the breakpoint with
probability $b/n$ and the cipher to the right with probability $1 - b/n$.)
If the resultant text after the swap has fewer bad words, then we keep it. Otherwise, we reject it.
This is done for $N_\text{ft} = 2000$ iterations for $A_\text{ft} = 4$ times.
We output the result of the best run in terms of the improvement in the number of valid words in the decoded plaintext.

## Results

I achieved first place on the leaderboard out of over 80 participants (students of the class) with an overall decoding accuracy of 99.83%
(no-breakpoint: 99.98%; breakpoint: 99.74%).

## Instructions

We use Python >=3.6 with NumPy.

While designing my algorithm, I made some of my own test cases from the passages given to us under [`data/texts/`](data/texts).
To run these test cases, run
```bash
python test.py
```
There are 89 test cases; each has 40 to 400 words.
Adding the `--short` option runs 909 test cases each with 4 to 39 words.
To see more options, run `python test.py -h`.

If you want to use my decoder to decode some of your own ciphertexts, use the decoder function located at [`src/decode.py`](src/decode.py):
```python3
decode(ciphertext: str, has_breakpoint: bool, debug_file_name: str = "test", debug: bool = False) -> str
```
where
* `ciphertext` is the input ciphertext
* `has_breakpoint` is whether there's a breakpoint in the encoding function
(location in the ciphertext where the cipher changes)
* `debug_file_name` is used in the name of the debugging log when `debug` is `True`
* `debug` specifies whether to print debug messages into the debugging log

Decoding a ciphertext usually takes ~15 seconds.
