# trains a character-level LSTM on tiny shakespeare, set up like train_gpt.py so the two
# can be compared: the same text and split, the same 16-character training windows in the
# same order, the same held-out windows, optimizer, lr schedule and number of steps, and
# about the same number of parameters (17,475 here, 17,505 in the gpt).
# the difference is how each model looks back. the gpt's attention looks at all earlier
# positions in the window at once. the lstm reads one character at a time and keeps what
# it needs in its state (h, c), so when it writes text it carries that state along and
# isn't limited to a 16-character window. 1500 steps take about 12 minutes
import os
import random
import time
import urllib.request

from engine import tensor, no_grad, stack
from nn import module, embedding, lstm_cell, linear, sample
from optim import AdamW, CosineAnnealingLR

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "datasets", "tinyshakespeare.txt")
WEIGHTS_PATH = os.path.join(HERE, "checkpoints", "char_lstm.json")

CONTEXT = 16      # characters in each training window
DIM = 32          # numbers per character going into the lstm
HIDDEN = 42       # size of h and c, picked so the parameter count matches the gpt
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


class char_lstm(module):
    # ids (B, T) -> scores (B, T, vocab) for the character after each position:
    #   x_t = embedding(ids[:, t])
    #   h_t, c_t = lstm_cell(x_t, (h_(t-1), c_(t-1)))
    #   scores_t = linear(h_t)
    # the scores at position t only depend on characters 0..t, because that's all the cell
    # has read by then, so unlike the gpt it needs no causal mask.
    # also returns the state after the last character, to carry on from it
    def __init__(self, vocab_size, dim, hidden):
        super().__init__()
        self.tok = embedding(vocab_size, dim)
        self.cell = lstm_cell(dim, hidden)
        self.head = linear(hidden, vocab_size)

    def __call__(self, ids, state=None):
        x = self.tok(tensor(ids))  # (B, T, dim)
        hs = []
        for t in range(len(ids[0])):
            state = self.cell(x[:, t], state)
            hs.append(state[0])
        return self.head(stack(hs, axis=1)), state


def get_batch(data, rng):
    # random windows of the text. the target for each position is the next character,
    # so one window gives CONTEXT guesses to learn from
    starts = [rng.randrange(len(data) - CONTEXT) for _ in range(BATCH)]
    inputs = [data[s:s + CONTEXT] for s in starts]
    targets = [c for s in starts for c in data[s + 1:s + CONTEXT + 1]]
    return inputs, targets


def loss_on(model, inputs, targets, state=None):
    # in training every window starts from a zero state, so the model sees the same 16
    # characters the gpt does
    scores, _ = model(inputs, state)
    return scores.reshape(len(inputs) * CONTEXT, -1).log_softmax().nll_loss(targets)


def evaluate(model, batches):
    with no_grad():
        return sum(loss_on(model, x, y).data[0] for x, y in batches) / len(batches)


def loss_after_reading(model, data, rng, before=48, batches=16):
    # training never runs the state past 16 characters, but writing text does. this checks
    # whether a longer run still helps: the model first reads the `before` characters that
    # come before each window, then scores the window starting from that state. returns
    # that loss and the loss on the same windows from a zero state
    after = zero = 0.0
    with no_grad():
        for _ in range(batches):
            starts = [rng.randrange(before, len(data) - CONTEXT) for _ in range(BATCH)]
            inputs = [data[s:s + CONTEXT] for s in starts]
            targets = [c for s in starts for c in data[s + 1:s + CONTEXT + 1]]
            _, state = model([data[s - before:s] for s in starts])
            after += loss_on(model, inputs, targets, state).data[0]
            zero += loss_on(model, inputs, targets).data[0]
    return after / batches, zero / batches


def generate(model, ids, length, temperature=0.8, top_k=None):
    # read the prompt once, then feed in each new character with the state from the step
    # before. the gpt reruns its whole window for every character; here each one costs a
    # single cell step
    ids = list(ids)
    with no_grad():
        scores, state = model([ids])
        for _ in range(length):
            ids.append(sample(scores[:, -1], temperature, top_k)[0])
            scores, state = model([ids[-1:]], state)
    return ids


random.seed(0)
text = load_text()
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
data = [char_to_id[c] for c in text]
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"{len(text)} characters, {len(chars)} different ones, {len(val_data)} held out\n")

model = char_lstm(len(chars), DIM, HIDDEN)
print(f"{sum(len(p.data) for p in model.parameters())} parameters\n")
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

after, zero = loss_after_reading(model, val_data, random.Random(4))
print(f"\nheld-out loss after first reading the 48 characters before each window {after:.3f},"
      f" from a zero state {zero:.3f}")

# save the weights, load them into a new model, and check it gives the same answers
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
model.save(WEIGHTS_PATH)
restored = char_lstm(len(chars), DIM, HIDDEN)
restored.load(WEIGHTS_PATH)
print(f"\nsaved to {os.path.relpath(WEIGHTS_PATH, HERE)}, reloaded model val loss {evaluate(restored, val_batches):.3f}")
