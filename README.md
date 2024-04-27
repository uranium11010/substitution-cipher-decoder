# Substitution cipher decoder

Final project for MIT 6.7800 (6.437).

Algorithm:
1. Use Metropolis-Hastings (MH) to sample from the posterior distribution of the cipher conditioned on the ciphertext.
2. Finetune by increasing the number of words that are in the list of the top 10000 most common words.
In each stage, several runs are conducted in parallel and the best one is kept.

Next step: Modify the finetuning stage to MH with negative log probability proportional to the sum of
Hamming distances between each word and the top-10000 word list.
