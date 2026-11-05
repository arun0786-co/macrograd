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
