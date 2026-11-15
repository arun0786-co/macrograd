import random
try:
    from .engine import tensor
except ImportError:
    # run as a script from this folder, not imported as a package
    from engine import tensor


def _rows(t):
    # how many examples t holds: the size of a tensor's first dim, or a list's length
    if isinstance(t, tensor):
        if not t.shape:
            raise ValueError("Cannot batch a tensor with no dims")
        return t.shape[0]
    return len(t)


def _take_rows(t, picks):
    # copies the rows at positions picks into a new tensor (or list). a tensor row is a
    # block of len(data) / rows values in the flat data, because the first dim is outermost
    if isinstance(t, tensor):
        size = len(t.data) // t.shape[0]
        data = [v for i in picks for v in t.data[i * size:(i + 1) * size]]
        return tensor(data, shape=(len(picks),) + t.shape[1:])
    return [t[i] for i in picks]


def batches(x, y, batch_size, shuffle=True):
    # splits a dataset into mini-batches. x and y hold one example per row (first dim):
    # x is a tensor of inputs, y a tensor of targets or a list of integer labels.
    # yields (x_batch, y_batch) pairs, with row i of x_batch still matching row i of y_batch.
    # with shuffle=True the rows come in a new random order each time it is called, so call
    # it once per epoch. the last batch is smaller when batch_size doesn't divide the rows.
    # the batches are new tensors, not connected to x or y in any graph
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size}")
    n = _rows(x)
    if _rows(y) != n:
        raise ValueError(f"x has {n} rows but y has {_rows(y)}")

    order = list(range(n))
    if shuffle:
        random.shuffle(order)
    for start in range(0, n, batch_size):
        picks = order[start:start + batch_size]
        yield _take_rows(x, picks), _take_rows(y, picks)


def _merge(ids, pair, new_id):
    # replaces every occurrence of the two ids in pair, side by side, with new_id, going left
    # to right, so in a run like a a a only the first two join
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class bpe:
    # a byte-pair encoding tokenizer (Sennrich et al., 2016), working on bytes like GPT-2's.
    # bpe(text, vocab_size) learns it from text: the text becomes its utf-8 bytes, ids 0-255,
    # then the most common pair of neighbouring ids becomes a new id, every occurrence is
    # replaced, and that repeats until there are vocab_size ids or no pair occurs twice.
    # on ties the pair seen first in the text wins. common words end up as one token and
    # rare ones are spelled with smaller pieces, so the same text takes fewer tokens than
    # characters, and since every byte is still a token, any text can be encoded, even
    # characters the training text never had.
    # encode applies the learned merges in the order they were learned. decode joins the
    # bytes of each id and reads them as utf-8, with a replacement character where the ids
    # cut a character's bytes apart.
    # it's plain python with no tensors, and training goes over the whole text once per
    # merge, so learn it from a slice of a big text
    def __init__(self, text, vocab_size):
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 256:
            raise ValueError(f"vocab_size must be an int >= 256, got {vocab_size!r}")
        ids = list(text.encode("utf-8"))
        self.merges = []  # merges[k] is the pair of ids that became id 256 + k
        self.vocab = [bytes([b]) for b in range(256)]  # vocab[i] is the bytes id i stands for
        while len(self.vocab) < vocab_size:
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            # max returns the first of equal counts, and the dict keeps the order pairs were seen in
            pair = max(counts, key=counts.get) if counts else None
            if pair is None or counts[pair] < 2:
                break
            ids = _merge(ids, pair, len(self.vocab))
            self.merges.append(pair)
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
        self._rank = {pair: k for k, pair in enumerate(self.merges)}

    def __len__(self):
        return len(self.vocab)

    def encode(self, text):
        # the earliest-learned pair that's in ids goes first. a merge never makes a pair
        # that was learned before it, so this is the same as applying every merge in order
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self._rank.get(p, len(self.merges)))
            if pair not in self._rank:
                break
            ids = _merge(ids, pair, 256 + self._rank[pair])
        return ids

    def decode(self, ids):
        for i in ids:
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.vocab):
                raise ValueError(f"{i!r} is not an id in 0..{len(self.vocab) - 1}")
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
