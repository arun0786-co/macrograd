# trains a tiny GPT that writes text one character at a time, on tiny shakespeare: about
# 1.1 million characters of shakespeare's plays in one text file. downloads on first run.
# the model sees 16 characters and learns to guess the next one at every position, then
# writes new text by guessing a character, adding it to the input and guessing again.
# 1500 steps take about 13 minutes. TOKENS = "bpe" makes it read and write byte-pair tokens
# instead of characters
import math
import os
import random
import time
import urllib.request

from engine import tensor, no_grad
from nn import module, embedding, transformer_block, layer_norm, rms_norm, linear, sample, sinusoidal_positions
from data import bpe
from optim import AdamW, CosineAnnealingLR, LinearWarmup

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "datasets", "tinyshakespeare.txt")
WEIGHTS_PATH = os.path.join(HERE, "checkpoints", "char_gpt.json")

CONTEXT = 16      # characters (or tokens) the model sees at once
DIM = 32          # numbers per character (or token) inside the model
HEADS = 4
LAYERS = 1
BATCH = 8
STEPS = 1500
# things to try. the defaults are what the numbers in the readme came from
WARMUP = 0              # steps of linear lr warmup before the cosine schedule, 0 for none
POSITIONS = "learned"   # "learned" position embedding, the fixed "sinusoidal" table, or "rope" in attention
LABEL_SMOOTHING = 0.0   # for the training loss only; the val loss is always plain nll
NORM = "layer"          # "layer" or "rms" norm, in the block and before the output layer
MLP = "gelu"            # "gelu" MLP or "swiglu" feed-forward in the block
TOKENS = "chars"        # "chars", or "bpe" tokens learned from the training text
MERGES = 132            # with TOKENS = "bpe": how many merges the tokenizer learns
EVAL_CHARS = 20000      # bits per character are measured on this many held-out characters


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
    # sin/cos table instead, which has nothing to learn, and positions="rope" adds nothing
    # to x and turns the queries and keys in attention instead.
    # norm and mlp pick the norm layers and the feed-forward. positions="rope", norm="rms"
    # and mlp="swiglu" together make it LLaMA-style
    def __init__(self, vocab_size, context, dim, num_heads, num_layers, positions="learned", norm="layer", mlp="gelu"):
        super().__init__()
        if positions not in ("learned", "sinusoidal", "rope"):
            raise ValueError(f"positions must be 'learned', 'sinusoidal' or 'rope', got {positions!r}")
        self.context = context
        self.dim = dim
        self.positions = positions
        self.tok = embedding(vocab_size, dim)
        if positions == "learned":
            self.pos = embedding(context, dim)
        self.blocks = [transformer_block(dim, num_heads, norm=norm, mlp=mlp, rope=positions == "rope")
                       for _ in range(num_layers)]
        self.ln = layer_norm(dim) if norm == "layer" else rms_norm(dim)
        self.head = linear(dim, vocab_size)

    def __call__(self, ids, caches=None):
        # caches holds one dict per block for generating with cached keys and values (see
        # multi_head_attention). only with positions="rope", since the other two add the
        # position to x and the new tokens' positions would have to be counted here
        if caches is not None and self.positions != "rope":
            raise ValueError("caches need positions='rope'")
        T = len(ids[0])
        x = self.tok(tensor(ids))
        # (B, T, dim) + (T, dim): every sequence in the batch gets the same positions
        if self.positions == "learned":
            x = x + self.pos(tensor(list(range(T))))
        elif self.positions == "sinusoidal":
            x = x + sinusoidal_positions(T, self.dim)
        for i, block in enumerate(self.blocks):
            x = block(x, True, None if caches is None else caches[i])
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


def generate(model, ids, length, temperature=0.8, top_k=None, use_cache=True):
    ids = list(ids)
    with no_grad():
        if model.positions != "rope" or not use_cache:
            for _ in range(length):
                # only the last CONTEXT characters fit, and only the scores at the last
                # position are about the character that comes next
                scores = model([ids[-model.context:]])[:, -1]
                ids.append(sample(scores, temperature, top_k)[0])
            return ids
        # with rope: read the prompt once, then run only the newest character each step.
        # each block keeps the keys and values of the CONTEXT - 1 characters before it, so
        # every character sees the same 16-character window as above. with one layer that's
        # exactly the same computation; with more, a later layer's cached keys came from
        # earlier windows, so it would only be close
        caches = [{} for _ in model.blocks]
        scores = model([ids[-model.context:]], caches)[:, -1]
        for _ in range(length):
            ids.append(sample(scores, temperature, top_k)[0])
            for cache in caches:
                cache["k"], cache["v"] = cache["k"][:, :, 1 - model.context:], cache["v"][:, :, 1 - model.context:]
            scores = model([ids[-1:]], caches)[:, -1]
    return ids


def bits_per_char(model, ids, num_chars):
    # the model's loss on a stretch of text in bits per character, which doesn't depend on
    # how the text was cut into tokens: the total -log2 p of every token after the first,
    # divided by the characters those tokens spell. windows of CONTEXT tokens step forward
    # by half a window, and each scores only the tokens the window before didn't, so every
    # token but the first few is guessed with at least half a window before it
    inputs, targets = ids[:-1], ids[1:]
    n, half = len(inputs), CONTEXT // 2
    ends = list(range(CONTEXT, n + 1, half))
    if ends[-1] != n:
        ends.append(n)
    windows, done = [], 0
    for end in ends:
        start = end - CONTEXT
        windows.append((start, done - start))  # the window, and its first new position
        done = end
    total = 0.0
    with no_grad():
        for b in range(0, len(windows), 32):
            group = windows[b:b + 32]
            log_probs = model([inputs[w:w + CONTEXT] for w, _ in group]).log_softmax()
            V = log_probs.shape[-1]
            for g, (w, first) in enumerate(group):
                for t in range(first, CONTEXT):
                    total -= log_probs.data[(g * CONTEXT + t) * V + targets[w + t]]
    return total / num_chars / math.log(2)


random.seed(0)
text = load_text()
split = int(0.9 * len(text))
train_text, val_text = text[:split], text[split:]
if TOKENS == "chars":
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    encode = lambda t: [char_to_id[c] for c in t]
    decode = lambda ids: "".join(chars[i] for i in ids)
    vocab_size = len(chars)
elif TOKENS == "bpe":
    start = time.time()
    tokenizer = bpe(train_text, 256 + MERGES)
    # only the bytes and merged tokens the training text uses get a row in the model, so no
    # weights go to the 191 bytes shakespeare never uses
    used = sorted(set(tokenizer.encode(train_text)))
    to_row = {t: i for i, t in enumerate(used)}
    encode = lambda t: [to_row[i] for i in tokenizer.encode(t)]
    decode = lambda ids: tokenizer.decode([used[i] for i in ids])
    vocab_size = len(used)
    print(f"learned {len(tokenizer.merges)} merges in {time.time() - start:.0f}s")
else:
    raise ValueError(f"TOKENS must be 'chars' or 'bpe', got {TOKENS!r}")
train_data, val_data = encode(train_text), encode(val_text)
eval_ids = encode(val_text[:EVAL_CHARS])
# the first token is never guessed, so its characters aren't counted
eval_chars = EVAL_CHARS - len(decode(eval_ids[:1]))
print(f"{len(text)} characters, {vocab_size} different {TOKENS}, {len(val_text) / len(val_data):.2f} characters "
      f"per token, {len(val_data)} tokens held out\n")

model = char_gpt(vocab_size, CONTEXT, DIM, HEADS, LAYERS, POSITIONS, NORM, MLP)
print(f"{sum(len(p.data) for p in model.parameters())} parameters")
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

print(f"held-out bits per character: {bits_per_char(model, eval_ids, eval_chars):.3f}")
prompt = encode("ROMEO:\n")
print("\n" + decode(generate(model, prompt, 300)))

# save the weights, load them into a new model, and check it gives the same answers
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
model.save(WEIGHTS_PATH)
restored = char_gpt(vocab_size, CONTEXT, DIM, HEADS, LAYERS, POSITIONS, NORM, MLP)
restored.load(WEIGHTS_PATH)
print(f"\nsaved to {os.path.relpath(WEIGHTS_PATH, HERE)}, reloaded model val loss {evaluate(restored, val_batches):.3f}")
