# trains a tiny GPT that writes text one character at a time, on tiny shakespeare: about
# 1.1 million characters of shakespeare's plays in one text file. downloads on first run.
# the model sees 16 characters and learns to guess the next one at every position, then
# writes new text by guessing a character, adding it to the input and guessing again.
# 1500 steps take about 13 minutes
import os
import random
import time
import urllib.request

from engine import tensor
from nn import module, embedding, transformer_block, layer_norm, linear, sample
from optim import AdamW, CosineAnnealingLR

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "datasets", "tinyshakespeare.txt")
WEIGHTS_PATH = os.path.join(HERE, "checkpoints", "char_gpt.json")

CONTEXT = 16      # characters the model sees at once
DIM = 32          # numbers per character inside the model
HEADS = 4
LAYERS = 1
BATCH = 8
STEPS = 1500


def load_text():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as response:
            text = response.read().decode()
        with open(DATA_PATH, "w") as f:
            f.write(text)
    with open(DATA_PATH) as f:
        return f.read()


class char_gpt(module):
    # ids (B, T) -> scores (B, T, vocab) for the character after each position:
    #   x = token_embedding(ids) + position_embedding(0..T-1)
    #   x = block(x), causal, for each block
    #   scores = linear(layer_norm(x))
    # the position embedding is how the model knows the order of the characters, since
    # attention on its own treats its input as a set
    def __init__(self, vocab_size, context, dim, num_heads, num_layers):
        super().__init__()
        self.context = context
        self.tok = embedding(vocab_size, dim)
        self.pos = embedding(context, dim)
        self.blocks = [transformer_block(dim, num_heads) for _ in range(num_layers)]
        self.ln = layer_norm(dim)
        self.head = linear(dim, vocab_size)

    def __call__(self, ids):
        T = len(ids[0])
        # (B, T, dim) + (T, dim): every sequence in the batch gets the same positions
        x = self.tok(tensor(ids)) + self.pos(tensor(list(range(T))))
        for block in self.blocks:
            x = block(x, causal=True)
        return self.head(self.ln(x))


def get_batch(data, rng):
    # random windows of the text. the target for each position is the next character,
    # so one window gives CONTEXT guesses to learn from
    starts = [rng.randrange(len(data) - CONTEXT) for _ in range(BATCH)]
    inputs = [data[s:s + CONTEXT] for s in starts]
    targets = [c for s in starts for c in data[s + 1:s + CONTEXT + 1]]
    return inputs, targets


def loss_on(model, inputs, targets):
    scores = model(inputs)
    return scores.reshape(len(inputs) * CONTEXT, -1).log_softmax().nll_loss(targets)


def evaluate(model, batches):
    return sum(loss_on(model, x, y).data[0] for x, y in batches) / len(batches)


def generate(model, ids, length, temperature=0.8, top_k=None):
    ids = list(ids)
    for _ in range(length):
        # only the last CONTEXT characters fit, and only the scores at the last position
        # are about the character that comes next
        scores = model([ids[-model.context:]])[:, -1]
        ids.append(sample(scores, temperature, top_k)[0])
    return ids


random.seed(0)
text = load_text()
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
data = [char_to_id[c] for c in text]
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"{len(text)} characters, {len(chars)} different ones, {len(val_data)} held out\n")

model = char_gpt(len(chars), CONTEXT, DIM, HEADS, LAYERS)
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
# there are no epochs here, so the schedule steps once per training step
schedule = CosineAnnealingLR(optimizer, T_max=STEPS)
val_rng = random.Random(1)
val_batches = [get_batch(val_data, val_rng) for _ in range(16)]
rng = random.Random(2)

start = time.time()
running = 0.0
for step in range(1, STEPS + 1):
    loss = loss_on(model, *get_batch(train_data, rng))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    schedule.step()
    running += loss.data[0]
    if step % 100 == 0:
        print(f"step {step} | train loss {running / 100:.3f} | val loss {evaluate(model, val_batches):.3f}"
              f" | {time.time() - start:.0f}s")
        running = 0.0

prompt = [char_to_id[c] for c in "ROMEO:\n"]
print("\n" + "".join(chars[i] for i in generate(model, prompt, 300)))

# save the weights, load them into a new model, and check it gives the same answers
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
model.save(WEIGHTS_PATH)
restored = char_gpt(len(chars), CONTEXT, DIM, HEADS, LAYERS)
restored.load(WEIGHTS_PATH)
print(f"\nsaved to {os.path.relpath(WEIGHTS_PATH, HERE)}, reloaded model val loss {evaluate(restored, val_batches):.3f}")
