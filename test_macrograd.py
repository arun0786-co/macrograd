import math
import random

from engine import tensor, gradcheck
from nn import MLP, dropout, linear
from optim import SGD


def close(a, b, rel=1e-9, abs_tol=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def raises(fn, error):
    try:
        fn()
    except error:
        return True
    return False


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


def rand_tensor(rows, cols):
    return tensor([[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)])


def test_core_op_grads():
    random.seed(1)
    x, w, b, y = rand_tensor(4, 3), rand_tensor(3, 2), rand_tensor(1, 2), rand_tensor(4, 2)
    gradcheck(lambda x, w, b: (x @ w + b).mse_loss(y), x, w, b)

    a = rand_tensor(3, 3)
    gradcheck(lambda a: ((a.relu() + a.leaky_relu()) ** 2).mean(), a)
    gradcheck(lambda a: (a * a + a).sum(), a)
    gradcheck(lambda a: (a.exp() + (a * a + 1.0).log()).sum(), a)
    gradcheck(lambda a: a.std() + (a / (a * a + 1.0)).sum(), a)


def test_mlp_grads():
    random.seed(2)
    model = MLP([3, 4, 4, 1], activation_fns=[lambda x: x.tanh(), lambda x: x.gelu()])
    x, y = rand_tensor(4, 3), rand_tensor(4, 1)
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())


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


def test_eval_turns_off_dropout():
    random.seed(0)
    model = MLP([3, 4, 4, 1], dropout_p=0.5)
    x = tensor([[1.0, 2.0, 3.0]])

    model.eval()
    assert model(x).data == model(x).data

    model.train()
    outputs = {tuple(model(x).data) for _ in range(20)}
    assert len(outputs) > 1


def test_dropout_graph_node():
    x = tensor([[1.0, 2.0, 3.0]])
    out = dropout(p=0.5)(x)
    assert out._op == 'dropout'
    assert x in out._prev
    assert all(child._op == '' for child in out._prev)


def test_linear_init_scales_with_fan_in():
    random.seed(0)
    layer = linear(400, 10)
    assert max(abs(w) for w in layer.w.data) <= 400 ** -0.5


def test_matmul_operator():
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0], [6.0]])
    assert (a @ b).data == [17.0, 39.0]
    assert (a @ b).shape == (2, 1)


def test_broadcast_either_side():
    m = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert (tensor(2.0) + m).data == (m + tensor(2.0)).data == [3.0, 4.0, 5.0, 6.0]
    assert (tensor([[10.0, 20.0]]) * m).data == [10.0, 40.0, 30.0, 80.0]

    s = tensor(3.0)
    row = tensor([[1.0, -2.0]])
    x = tensor([[0.5, 1.0], [-1.0, 2.0]])
    gradcheck(lambda a, b, c: ((a + c) * (b * c)).sum(), s, row, x)


def test_invalid_log_and_pow_inputs_raise():
    assert raises(lambda: tensor([-1.0]).log(), ValueError)
    assert raises(lambda: tensor([-4.0]) ** 0.5, ValueError)
    assert (tensor([-2.0]) ** 3).data == [-8.0]
    assert close(tensor([0.0]).log().data[0], math.log(1e-8))



def test_tensor_input_checks():
    m = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert raises(lambda: tensor([1.0, 2.0, 3.0], shape=(2, 2)), ValueError)
    assert raises(lambda: tensor([1.0, [2.0, 3.0]]), ValueError)
    assert raises(lambda: m.get_element(0, 3), IndexError)
    assert raises(lambda: m.get_element(1), IndexError)
    assert m.get_element(1, 0) == 3.0
    assert tensor([1.0, 2.0], shape=[2]).shape == (2,)


def test_std_of_constant_has_zero_grad():
    x = tensor([2.0, 2.0, 2.0])
    out = x.std()
    out.backward()
    assert out.data == [0.0]
    assert x.grad == [0.0, 0.0, 0.0]


def test_backward_on_deep_graph():
    x = tensor([1.0])
    y = x
    for _ in range(5000):
        y = y * 1.0 + 0.0
    y.backward()
    assert x.grad == [1.0]


def test_sgd_training_step():
    random.seed(3)
    model = MLP([2, 3, 1])
    opt = SGD(model.parameters(), lr=0.1)
    x, y = tensor([[1.0, 2.0]]), tensor([[0.5]])
    losses = []
    for _ in range(20):
        loss = model(x).mse_loss(y)
        opt.zero_grad()
        assert all(g == 0.0 for p in model.parameters() for g in p.grad)
        loss.backward()
        opt.step()
        losses.append(loss.data[0])
    assert losses[-1] < losses[0]


def test_mlp_and_dropout_argument_checks():
    relu = lambda x: x.relu()
    assert raises(lambda: MLP([3, 4, 4, 1], activation_fns=[relu]), ValueError)
    assert raises(lambda: MLP([3, 4, 4, 1], dropout_p=[0.2]), ValueError)
    assert raises(lambda: dropout(p=1.0), ValueError)


def test_mlp_output_activation():
    random.seed(4)
    model = MLP([2, 3, 1], activation_fns=[lambda x: x.relu(), lambda x: x.tanh()])
    x = tensor([[0.5, -1.0]])
    expected = model.layers[1](model.layers[0](x).relu()).tanh()
    assert model(x).data == expected.data



def test_gradcheck_catches_wrong_backward():
    def bad_square(a):
        out = tensor([v * v for v in a.data], shape=a.shape, children=(a,), op='bad_square')

        def _backward():
            for i in range(len(a.data)):
                a.grad[i] += a.data[i] * out.grad[i]  # should be 2 * a.data[i]

        out._backward = _backward
        return out.sum()

    x = tensor([1.0, 2.0, -3.0])
    assert raises(lambda: gradcheck(bad_square, x), ValueError)
    assert gradcheck(lambda a: (a * a).sum(), x)
    assert raises(lambda: gradcheck(lambda a: a * a, x), ValueError)



def test_reshape():
    a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.reshape(3, 2)
    assert b.shape == (3, 2) and b.data == a.data
    assert b.get_element(2, 1) == 6.0
    assert a.shape == (2, 3)

    assert a.reshape(-1).shape == (6,)
    assert a.reshape(3, -1).shape == (3, 2)
    assert a.reshape((1, -1)).shape == (1, 6)
    assert tensor([5.0]).reshape(()).shape == ()

    assert raises(lambda: a.reshape(4, 2), ValueError)
    assert raises(lambda: a.reshape(-1, -1), ValueError)
    assert raises(lambda: a.reshape(5, -1), ValueError)
    assert raises(lambda: a.reshape(-2, -3), ValueError)

    # distinct weights, so a gradient landing in the wrong position would show up
    w = tensor([[1.0], [-2.0]])
    weights = tensor([0.5, -1.0, 2.0, 3.0, -0.7, 1.5])
    gradcheck(lambda x: (x.reshape(3, 2) @ w).sum() + (x.reshape(-1) * weights).sum(), a)


def test_transpose():
    a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = a.transpose()
    assert t.shape == (3, 2) and t.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    assert a.transpose(0, 1).data == a.transpose(1, 0).data == t.data
    assert t.transpose().data == a.data
    assert a.transpose(0, 0).data == a.data

    # (A B)^T = B^T A^T
    b = tensor([[1.0, -1.0], [0.5, 2.0], [3.0, 0.0]])
    assert (a @ b).transpose().data == (b.transpose() @ a.transpose()).data

    x = tensor([float(v) for v in range(24)], shape=(2, 3, 4))
    y = x.transpose(0, 2)
    assert y.shape == (4, 3, 2)
    assert x.transpose(-1, 0).data == y.data
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert y.get_element(k, j, i) == x.get_element(i, j, k)
                assert x.transpose().get_element(i, k, j) == x.get_element(i, j, k)

    assert raises(lambda: tensor([1.0, 2.0]).transpose(), IndexError)
    assert raises(lambda: x.transpose(0, 3), IndexError)

    # distinct weights, so a gradient landing in the wrong position would show up
    w = tensor([[1.0], [-2.0]])
    weights = tensor([[0.5, -1.0], [2.0, 3.0], [-0.7, 1.5]])
    gradcheck(lambda m: (m.transpose() @ w).sum() + (m.transpose() * weights).sum(), a)

    weights = tensor([0.5 * v - 3.0 for v in range(24)], shape=(4, 3, 2))
    gradcheck(lambda m: (m.transpose(0, 2) * weights).sum(), x)


def test_sum_over_axis():
    a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert a.sum().data == [21.0] and a.sum().shape == ()
    assert a.sum(keepdim=True).shape == (1, 1)
    assert a.sum(axis=0).data == [5.0, 7.0, 9.0] and a.sum(axis=0).shape == (3,)
    assert a.sum(axis=1).data == [6.0, 15.0] and a.sum(axis=1).shape == (2,)
    assert a.sum(axis=-1).data == a.sum(axis=1).data
    assert a.sum(axis=0, keepdim=True).shape == (1, 3)
    assert a.sum(axis=1, keepdim=True).shape == (2, 1)

    x = tensor([float(v) for v in range(24)], shape=(2, 3, 4))
    for axis, shape in [(0, (3, 4)), (1, (2, 4)), (2, (2, 3))]:
        s = x.sum(axis=axis)
        assert s.shape == shape
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    out_index = [i, j, k]
                    del out_index[axis]
                    expected = sum(x.get_element(*(out_index[:axis] + [t] + out_index[axis:]))
                                   for t in range(x.shape[axis]))
                    assert s.get_element(*out_index) == expected

    assert raises(lambda: a.sum(axis=2), IndexError)
    assert raises(lambda: tensor(3.0).sum(axis=0), IndexError)

    # distinct weights, so a gradient landing in the wrong position would show up
    for axis in [0, 1, 2]:
        shape = x.sum(axis=axis).shape
        size = len(x.sum(axis=axis).data)
        weights = tensor([0.7 * v - 2.0 for v in range(size)], shape=shape)
        gradcheck(lambda m: (m.sum(axis=axis) * weights).sum(), x)
    w = tensor([[1.0], [-2.0], [0.5]])
    gradcheck(lambda m: (m.sum(axis=0, keepdim=True) @ w).sum(), a)


def test_full_broadcasting():
    col = tensor([[1.0], [2.0], [3.0]])
    row = tensor([10.0, 20.0, 30.0, 40.0])
    out = col + row
    assert out.shape == (3, 4)
    for i in range(3):
        for j in range(4):
            assert out.get_element(i, j) == col.get_element(i, 0) + row.get_element(j)
    assert (row * col).data == (col * row).data

    x = tensor([float(v) for v in range(24)], shape=(2, 3, 4))
    m = tensor([[2.0], [-1.0], [0.5]])
    out = x * m
    assert out.shape == (2, 3, 4)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert out.get_element(i, j, k) == x.get_element(i, j, k) * m.get_element(j, 0)

    e = tensor([[1.0, 2.0]]).expand((2, 3, 2))
    assert e.shape == (2, 3, 2) and e.data == [1.0, 2.0] * 6
    assert tensor(5.0).expand((2, 2)).data == [5.0] * 4

    assert raises(lambda: tensor([1.0, 2.0]) + tensor([1.0, 2.0, 3.0]), ValueError)
    assert raises(lambda: tensor([[1.0, 2.0]]).expand((2,)), ValueError)
    assert raises(lambda: tensor([[1.0, 2.0], [3.0, 4.0]]).expand((3, 2)), ValueError)

    # distinct weights, so a grad added to the wrong value would show up
    weights = tensor([0.3 * v - 1.0 for v in range(24)], shape=(2, 3, 4))
    gradcheck(lambda a, b: ((a + b) * weights).sum(), col, row)
    gradcheck(lambda a, b: (a * b * weights).sum(), x, m)
    gradcheck(lambda a: (a.expand((2, 3, 4)) * weights).sum(), tensor([[0.5], [-1.0], [2.0]]))


def test_mean_and_var_over_axis():
    a = tensor([[1.0, 2.0, 6.0], [4.0, 4.0, 4.0]])
    assert close(a.mean().data[0], 21.0 / 6)
    assert a.mean(axis=0).data == [2.5, 3.0, 5.0] and a.mean(axis=0).shape == (3,)
    assert a.mean(axis=1).data == [3.0, 4.0] and a.mean(axis=-1, keepdim=True).shape == (2, 1)

    v = a.var(axis=1)
    assert v.shape == (2,)
    assert close(v.data[0], ((1 - 3) ** 2 + (2 - 3) ** 2 + (6 - 3) ** 2) / 3) and v.data[1] == 0.0
    assert a.var(axis=0, keepdim=True).shape == (1, 3)
    assert close(a.var(axis=0).data[2], 1.0)
    values = a.data
    m = sum(values) / 6
    assert close(a.var().data[0], sum((x - m) ** 2 for x in values) / 6)

    x = tensor([0.1 * v * v - v for v in range(24)], shape=(2, 3, 4))
    for axis in [0, 1, 2]:
        out = x.mean(axis=axis)
        weights = tensor([0.5 * v - 2.0 for v in range(len(out.data))], shape=out.shape)
        gradcheck(lambda t: (t.mean(axis=axis) * weights).sum() + (t.var(axis=axis) * weights).sum(), x)

    assert raises(lambda: a.mean(axis=2), IndexError)


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
