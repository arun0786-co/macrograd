import math
import random

from engine import tensor


def close(a, b, rel=1e-9, abs_tol=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def gradcheck(f, *inputs, eps=1e-6, tol=1e-6):
    # compare backward() against central finite differences
    f(*inputs).backward()
    for x in inputs:
        for i in range(len(x.data)):
            old = x.data[i]
            x.data[i] = old + eps
            up = f(*inputs).data[0]
            x.data[i] = old - eps
            down = f(*inputs).data[0]
            x.data[i] = old
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - x.grad[i]) < tol, (i, numeric, x.grad[i])


def ref_sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    return math.exp(x) / (1 + math.exp(x))


def test_exp_matches_math():
    for x in [-700, -50, -10, -5, -1, -1e-3, 0, 1e-3, 1, 5, 10, 50, 700]:
        assert close(tensor._taylor_exp(x), math.exp(x)), x
    assert tensor._taylor_exp(1000) == float('inf')
    assert tensor._taylor_exp(-1000) == 0.0


def test_log_matches_math():
    for x in [1e-300, 1e-8, 1e-4, 0.01, 0.5, 1.0, 2.0, 100.0, 1e4, 1e300]:
        assert close(tensor._taylor_log(x), math.log(x)), x


def test_sigmoid_and_tanh_values():
    for x in [-1000.0, -20.0, -1.0, 0.0, 1.0, 20.0, 1000.0]:
        assert close(tensor([x]).sigmoid().data[0], ref_sigmoid(x)), x
        assert close(tensor([x]).tanh().data[0], math.tanh(x)), x


def test_sigmoid_and_tanh_grads():
    x = tensor([-3.0, -0.5, 0.0, 0.7, 4.0])
    gradcheck(lambda a: (a.sigmoid() + a.tanh()).sum(), x)

    big = tensor([-1000.0, 1000.0])
    (big.sigmoid() + big.tanh()).sum().backward()
    assert big.grad == [0.0, 0.0]


def test_gelu():
    k = math.sqrt(2 / math.pi)
    for x in [-10.0, -3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0]:
        out = tensor([x]).gelu().data[0]
        assert close(out, 0.5 * x * (1 + math.tanh(k * (x + 0.044715 * x ** 3)))), x
        # the tanh form stays close to the exact erf form
        assert abs(out - 0.5 * x * (1 + math.erf(x / math.sqrt(2)))) < 1e-3, x
    gradcheck(lambda a: a.gelu().sum(), tensor([-3.0, -1.0, -0.2, 0.0, 0.4, 1.5, 3.0]))


def test_softmax_per_row():
    logits = [[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]
    out = tensor(logits).softmax()
    for r, row in enumerate(logits):
        m = max(row)
        denom = sum(math.exp(v - m) for v in row)
        for j, v in enumerate(row):
            assert close(out.data[r * 3 + j], math.exp(v - m) / denom), (r, j)

    weights = tensor([[0.3, -1.0, 2.0], [1.5, 0.2, -0.7]])
    x = tensor([[0.5, -0.3, 0.1], [2.0, 1.0, -1.0]])
    gradcheck(lambda a: (a.softmax() * weights).sum(), x)


def ref_cross_entropy(logits, targets):
    total = 0.0
    for row, t in zip(logits, targets):
        m = max(row)
        log_sum_exp = m + math.log(sum(math.exp(v - m) for v in row))
        total -= sum(tj * (v - log_sum_exp) for v, tj in zip(row, t))
    return total / len(logits)


def test_cross_entropy_values():
    # confidently wrong prediction
    loss = tensor([0.0, 12.0]).cross_entropy_loss(tensor([1.0, 0.0])).data[0]
    assert close(loss, ref_cross_entropy([[0.0, 12.0]], [[1.0, 0.0]]))

    logits = [[5.0, 0.0], [5.0, 0.0], [-2.0, 40.0]]
    targets = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    loss = tensor(logits).cross_entropy_loss(tensor(targets)).data[0]
    assert close(loss, ref_cross_entropy(logits, targets))


def test_cross_entropy_grads():
    targets = tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    x = tensor([[0.3, -0.2, 0.5], [2.0, 0.1, -1.0]])
    gradcheck(lambda a: a.cross_entropy_loss(targets), x)


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
