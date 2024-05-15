import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.constants import ALPHABET_SIZE, ALPHABET
from src.attention_model import AttentionModel
from src.dataloader import PlaintextDataset, PlaintextDataloader


def do_test(model, dataloader):
    with torch.no_grad():
        num_texts = 0
        num_correct = 0
        for b_idx, batch in enumerate(dataloader):
            num_texts += len(batch)
            logits = model(batch)
            preds = logits.argmax(-1)
            num_correct += torch.sum(torch.gather(preds, 1, batch.long()) == batch.long())
    return num_correct / (num_texts * batch.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    torch.set_default_device('cuda')

    train_file = "data/train.txt"
    train_dataset = PlaintextDataset(train_file)
    train_dataloader = PlaintextDataloader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    print("TRAIN SET SIZE:", len(train_dataset))
    valid_file = "data/valid.txt"
    valid_dataset = PlaintextDataset(valid_file)
    valid_dataloader = PlaintextDataloader(valid_dataset, batch_size=args.batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
    print("VALID SET SIZE:", len(valid_dataset))
    test_file = "data/test.txt"
    test_dataset = PlaintextDataset(test_file)
    test_dataloader = PlaintextDataloader(test_dataset, batch_size=args.batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
    print("TEST SET SIZE:", len(test_dataset))

    model = AttentionModel(args.hidden_dim, args.num_layers, args.num_heads)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    alphabet_inds = torch.arange(ALPHABET_SIZE)

    for epoch in range(100000):
        model.train()
        num_texts = 0
        num_correct = 0
        for b_idx, train_batch in enumerate(train_dataloader):
            num_texts += len(train_batch)
            optimizer.zero_grad()
            logits = model(train_batch)
            labels = alphabet_inds.broadcast_to(logits.size()[:2])
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1))
            loss.backward()
            optimizer.step()
            preds = logits.argmax(-1)
            num_correct += torch.sum(torch.gather(preds, 1, train_batch.long()) == train_batch.long())
        if (epoch + 1) % 100 == 0:
            print(''.join(ALPHABET), ''.join(ALPHABET), ''.join(ALPHABET))
            print(' '.join(''.join([ALPHABET[idx] for idx in preds[i]]) for i in range(-3, 0)))
            train_acc = num_correct / (num_texts * train_batch.shape[1])
            model.eval()
            valid_acc = do_test(model, valid_dataloader)
            print(f"EPOCH {epoch} TRAIN ACC: {train_acc:.6f}; VALID ACC: {valid_acc:.6f}")
    test_acc = do_test(model, test_dataloader)
    print(f"TEST ACC: {test_acc:.6f}")
