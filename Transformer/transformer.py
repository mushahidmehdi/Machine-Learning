
import torch
from torch.nn import functional as F
import torch.nn as nn


block_sz = 4
context_sz = 8


# opeing the file after downloading the data from tinny Shakespearean and reading

with open('file.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_len = len(chars)

# mapping all the unique char in the data agint their index.
stoi = {ch: idx for idx, ch in enumerate(chars)}
itos = {idx: ch for idx, ch in enumerate(chars)}


def encoder(st): return [stoi[i] for i in st]
def decoder(it): return ''.join([itos[i] for i in it])


# changing data into a tensor.
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9 * vocab_len)
train_data = data[:n]
val_data = data[n:]


def get_batch(dt):
    data = train_data if dt == 'train' else val_data
    ix = torch.randint(len(data) - context_sz, (block_sz,))
    x = torch.stack([data[i: i + context_sz] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + context_sz] for i in ix])
    return x, y


xb, yb = get_batch('train')


class BigramLanguageModel(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.embedding_table = nn.Embedding(tab_dim, tab_dim)

    def forward(self, idx, target=None):
        logits = self.embedding_table(idx)
        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def genrate(self, idx, max_token):
        for _ in range(max_token):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


m = BigramLanguageModel(vocab_len)
logits, loss = m(xb, yb)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


context_sz = 32
for _ in range(10000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
