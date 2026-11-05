# trains a small CNN to read handwritten digits, using the UCI optdigits dataset: 8x8
# images where each pixel counts the ink in a 4x4 block of the original scan (0 to 16).
# 3823 training images from 30 people and 1797 test images from 13 other people, so the
# test score shows how well it reads handwriting it has never seen. downloads on first run
import os
import random
import time
import urllib.request

from engine import tensor
from nn import sequential, conv2d, max_pool2d, flatten, linear, kaiming_normal
from optim import AdamW, CosineAnnealingLR
from data import batches

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "datasets")
WEIGHTS_PATH = os.path.join(HERE, "checkpoints", "digits_cnn.json")


def load_digits(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {URL + filename}")
        with urllib.request.urlopen(URL + filename, timeout=60) as response:
            text = response.read().decode()
        with open(path, "w") as f:
            f.write(text)
    images, labels = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            values = [int(v) for v in line.split(",")]
            images += [v / 16.0 for v in values[:64]]
            labels.append(values[64])
    # one channel per image: (N, 1, 8, 8)
    return tensor(images, shape=(len(labels), 1, 8, 8)), labels


def make_model():
    # 8x8 -> conv -> 8x8 -> pool -> 4x4 -> conv -> 4x4 -> pool -> 2x2, then 16 * 2 * 2 = 64
    # numbers per image go into a linear layer with one score per digit
    model = sequential(
        conv2d(1, 8, kernel_size=3, padding=1), lambda x: x.relu(), max_pool2d(2),
        conv2d(8, 16, kernel_size=3, padding=1), lambda x: x.relu(), max_pool2d(2),
        flatten(), linear(64, 10),
    )
    # the conv layers feed relus, so use kaiming init for their filters
    for layer in model.layers:
        if isinstance(layer, conv2d):
            kaiming_normal(layer.w)
    return model


def evaluate(model, images, labels):
    # in chunks, so the graph for all test images isn't held at once
    correct, loss_sum = 0, 0.0
    for xb, yb in batches(images, labels, batch_size=256, shuffle=False):
        scores = model(xb)
        loss_sum += scores.log_softmax().nll_loss(yb).data[0] * len(yb)
        correct += round(scores.accuracy(yb) * len(yb))
    return loss_sum / len(labels), correct / len(labels)


random.seed(0)
train_x, train_y = load_digits("optdigits.tra")
test_x, test_y = load_digits("optdigits.tes")
print(f"{len(train_y)} training images, {len(test_y)} test images\n")

model = make_model()
epochs = 5
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
schedule = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(1, epochs + 1):
    start = time.time()
    loss_sum, seen = 0.0, 0
    for xb, yb in batches(train_x, train_y, batch_size=32):
        loss = model(xb).log_softmax().nll_loss(yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.data[0] * len(yb)
        seen += len(yb)
    schedule.step()
    test_loss, test_acc = evaluate(model, test_x, test_y)
    print(f"epoch {epoch} | train loss {loss_sum / seen:.4f} | test loss {test_loss:.4f}"
          f" acc {test_acc:.4f} | {time.time() - start:.0f}s")

# save the weights, load them into a new model, and check it gives the same answers
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
model.save(WEIGHTS_PATH)
restored = make_model()
restored.load(WEIGHTS_PATH)
_, restored_acc = evaluate(restored, test_x, test_y)
print(f"\nsaved to {os.path.relpath(WEIGHTS_PATH, HERE)}, reloaded model test acc {restored_acc:.4f}")
