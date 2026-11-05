# trains an MLP to tell apart three kinds of iris flower from four measurements
# (sepal length and width, petal length and width), using the classic UCI iris dataset:
# 150 flowers, 50 of each kind. downloads it on the first run.
import os
import random
import urllib.request

from engine import tensor
from nn import MLP
from optim import AdamW, CosineAnnealingLR
from data import batches

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "datasets", "iris.data")
WEIGHTS_PATH = os.path.join(HERE, "checkpoints", "iris_mlp.json")
CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_iris():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=30) as response:
            text = response.read().decode()
        with open(DATA_PATH, "w") as f:
            f.write(text)
    features, labels = [], []
    with open(DATA_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            *values, name = line.strip().split(",")
            features.append([float(v) for v in values])
            labels.append(CLASSES.index(name))
    return features, labels


def split(features, labels, test_per_class=10):
    # same number of each kind in the test set, so it isn't lopsided by chance
    train, test = [], []
    for c in range(len(CLASSES)):
        rows = [i for i, label in enumerate(labels) if label == c]
        random.shuffle(rows)
        test += rows[:test_per_class]
        train += rows[test_per_class:]
    pick = lambda rows: ([features[i] for i in rows], [labels[i] for i in rows])
    return pick(train), pick(test)


def standardize(train_x, test_x):
    # scale each measurement to mean 0, std 1 using the training rows only, so the test
    # rows stay unseen. without this, petal length (1-7 cm) would swamp petal width (0.1-2.5 cm)
    n, cols = len(train_x), len(train_x[0])
    means = [sum(row[j] for row in train_x) / n for j in range(cols)]
    stds = [(sum((row[j] - means[j]) ** 2 for row in train_x) / n) ** 0.5 for j in range(cols)]
    scale = lambda rows: [[(row[j] - means[j]) / stds[j] for j in range(cols)] for row in rows]
    return scale(train_x), scale(test_x)


def evaluate(model, x, labels):
    model.eval()
    scores = model(x)
    loss = scores.log_softmax().nll_loss(labels).data[0]
    model.train()
    return loss, scores.accuracy(labels)


random.seed(0)
features, labels = load_iris()
(train_x, train_y), (test_x, test_y) = split(features, labels)
train_x, test_x = standardize(train_x, test_x)
train_x, test_x = tensor(train_x), tensor(test_x)
print(f"{len(train_y)} training flowers, {len(test_y)} test flowers\n")

model = MLP([4, 16, 3])
epochs = 60
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
schedule = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(1, epochs + 1):
    for xb, yb in batches(train_x, train_y, batch_size=16):
        loss = model(xb).log_softmax().nll_loss(yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    schedule.step()

    if epoch == 1 or epoch % 10 == 0:
        train_loss, train_acc = evaluate(model, train_x, train_y)
        test_loss, test_acc = evaluate(model, test_x, test_y)
        print(f"epoch {epoch:<3} lr {optimizer.lr:.4f} | train loss {train_loss:.4f} acc {train_acc:.3f}"
              f" | test loss {test_loss:.4f} acc {test_acc:.3f}")

# save the weights, load them into a new model, and check it gives the same answers
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
model.save(WEIGHTS_PATH)
restored = MLP([4, 16, 3])
restored.load(WEIGHTS_PATH)
_, restored_acc = evaluate(restored, test_x, test_y)
print(f"\nsaved to {os.path.relpath(WEIGHTS_PATH, HERE)}, reloaded model test acc {restored_acc:.3f}")
