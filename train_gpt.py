# trains a tiny GPT that writes text one character at a time, on tiny shakespeare: about
# 1.1 million characters of shakespeare's plays in one text file. downloads on first run.
# the model sees 16 characters and learns to guess the next one at every position, then
# writes new text by guessing a character, adding it to the input and guessing again.
# 1500 steps take about 13 minutes
import os
import random
import time
import urllib.request

from engine import tensor, no_grad
from nn import module, embedding, transformer_block, layer_norm, linear, sample, sinusoidal_positions
from optim import AdamW, CosineAnnealingLR, LinearWarmup

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
# three things to try. the defaults are what the numbers in the readme came from
WARMUP = 0              # steps of linear lr warmup before the cosine schedule, 0 for none
POSITIONS = "learned"   # "learned" position embedding or the fixed "sinusoidal" table
LABEL_SMOOTHING = 0.0   # for the training loss only; the val loss is always plain nll


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
    # attention on its own treats its input as a set. positions="sinusoidal" adds the fixed
    # sin/cos table instead, which has nothing to learn
    def __init__(self, vocab_size, context, dim, num_heads, num_layers, positions="learned"):
        super().__init__()
        if positions not in ("learned", "sinusoidal"):
            raise ValueError(f"positions must be 'learned' or 'sinusoidal', got {positions!r}")
        self.context = context
        self.dim = dim
        self.positions = positions
        self.tok = embedding(vocab_size, dim)
        if positions == "learned":
            self.pos = embedding(context, dim)
        self.blocks = [transformer_block(dim, num_heads) for _ in range(num_layers)]
        self.ln = layer_norm(dim)
        self.head = linear(dim, vocab_size)

    def __call__(self, ids):
        T = len(ids[0])
        if self.positions == "learned":
            pos = self.pos(tensor(list(range(T))))
        else:
            pos = sinusoidal_positions(T, self.dim)
        # (B, T, dim) + (T, dim): every sequence in the batch gets the same positions
        x = self.tok(tensor(ids)) + pos
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


def loss_on(model, inputs, targets, label_smoothing=0.0):
    scores = model(inputs)
    return scores.reshape(len(inputs) * CONTEXT, -1).log_softmax().nll_loss(targets, label_smoothing)


def evaluate(model, batches):
    with no_grad():
        return sum(loss_on(model, x, y).data[0] for x, y in batches) / len(batches)


def generate(model, ids, length, temperature=0.8, top_k=None):
    ids = list(ids)
    with no_grad():
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

model = char_gpt(len(chars), CONTEXT, DIM, HEADS, LAYERS, POSITIONS)
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
# there are no epochs here, so the schedule steps once per training step
if WARMUP:
    schedule = LinearWarmup(optimizer, WARMUP, after=CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP))
else:
    schedule = CosineAnnealingLR(optimizer, T_max=STEPS)
val_rng = random.Random(1)
val_batches = [get_batch(val_data, val_rng) for _ in range(16)]
rng = random.Random(2)

start = time.time()
running = 0.0
for step in range(1, STEPS + 1):
    loss = loss_on(model, *get_batch(train_data, rng), LABEL_SMOOTHING)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    schedule.step()
    # with LABEL_SMOOTHING this is the smoothed loss, which reads higher than plain nll
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
restored = char_gpt(len(chars), CONTEXT, DIM, HEADS, LAYERS, POSITIONS)
restored.load(WEIGHTS_PATH)
print(f"\nsaved to {os.path.relpath(WEIGHTS_PATH, HERE)}, reloaded model val loss {evaluate(restored, val_batches):.3f}")
