import json
import math
import os
import random
import tempfile

from engine import tensor, concat, gradcheck
from nn import MLP, batch_norm, dropout, embedding, layer_norm, linear, module, sequential
from nn import xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal
from optim import SGD, Adam, AdamW, clip_grad_norm, StepLR, CosineAnnealingLR
from data import batches


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


def test_max_and_argmax():
    a = tensor([[1.0, 7.0, 3.0], [9.0, -2.0, 9.0]])
    assert a.max().data == [9.0] and a.max().shape == ()
    assert a.argmax().data == [3]
    assert a.max(axis=1).data == [7.0, 9.0] and a.argmax(axis=1).data == [1, 0]
    assert a.max(axis=0).data == [9.0, 7.0, 9.0] and a.argmax(axis=0).data == [1, 0, 1]
    assert a.max(axis=-1, keepdim=True).shape == (2, 1)
    assert a.argmax(axis=0, keepdim=True).shape == (1, 3)

    random.seed(5)
    x = tensor([random.uniform(-1, 1) for _ in range(24)], shape=(2, 3, 4))
    for axis in [0, 1, 2]:
        m, idx = x.max(axis=axis), x.argmax(axis=axis)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    out_index = [i, j, k]
                    del out_index[axis]
                    column = [x.get_element(*(out_index[:axis] + [t] + out_index[axis:]))
                              for t in range(x.shape[axis])]
                    assert m.get_element(*out_index) == max(column)
                    assert idx.get_element(*out_index) == column.index(max(column))

    # ties: the grad goes only to the first max, the one argmax points at
    t = tensor([2.0, 5.0, 5.0])
    t.max().backward()
    assert t.grad == [0.0, 1.0, 0.0]

    assert raises(lambda: a.max(axis=2), IndexError)
    assert raises(lambda: tensor([], shape=(0,)).max(), ValueError)

    for axis in [None, 0, 1, 2]:
        out = x.max(axis=axis)
        weights = tensor([0.5 * v - 2.0 for v in range(len(out.data))], shape=out.shape)
        gradcheck(lambda t: (t.max(axis=axis) * weights).sum(), x)


def test_getitem():
    a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert a[1].data == [4.0, 5.0, 6.0] and a[1].shape == (3,)
    assert a[-1].data == a[1].data
    assert a[0, 2].data == [3.0] and a[0, 2].shape == ()
    assert a[:, 1].data == [2.0, 5.0] and a[:, 1].shape == (2,)
    assert a[:, ::2].data == [1.0, 3.0, 4.0, 6.0] and a[:, ::2].shape == (2, 2)
    assert a[0:1].shape == (1, 3)
    assert a[:, ::-1].data == [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    assert a[[1, 0, 1]].data == [4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert a[5:].shape == (0, 3)

    x = tensor([float(v) for v in range(24)], shape=(2, 3, 4))
    y = x[1, 1:, [3, 0]]
    assert y.shape == (2, 2)
    for j in range(2):
        for k, c in enumerate([3, 0]):
            assert y.get_element(j, k) == x.get_element(1, j + 1, c)

    assert raises(lambda: a[2], IndexError)
    assert raises(lambda: a[0, 0, 0], IndexError)
    assert raises(lambda: a[[0, 3]], IndexError)
    assert raises(lambda: a[0.5], TypeError)

    # a repeated row gets its grad added twice
    rows = a[[1, 1]]
    rows.sum().backward()
    assert a.grad == [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]

    weights = tensor([[0.5, -1.0], [2.0, 3.0], [-0.7, 1.5]])
    gradcheck(lambda t: (t[1, :, [3, 0, 3]] * tensor([0.5, -1.0, 2.0])).sum()
              + (t[:, 1:, 0].transpose() * weights[:2]).sum(), x)


def test_concat():
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0]])
    c = tensor([[7.0], [8.0]])
    assert concat([a, b]).data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] and concat([a, b]).shape == (3, 2)
    assert concat([a, c], axis=1).data == [1.0, 2.0, 7.0, 3.0, 4.0, 8.0]
    assert concat([a, c], axis=-1).shape == (2, 3)
    assert concat([tensor([1.0]), tensor([2.0, 3.0])]).data == [1.0, 2.0, 3.0]

    x = tensor([float(v) for v in range(24)], shape=(2, 3, 4))
    y = tensor([100.0 + v for v in range(16)], shape=(2, 2, 4))
    z = concat([x, y], axis=1)
    assert z.shape == (2, 5, 4)
    for i in range(2):
        for j in range(5):
            for k in range(4):
                expected = x.get_element(i, j, k) if j < 3 else y.get_element(i, j - 3, k)
                assert z.get_element(i, j, k) == expected
    # concat undoes slicing
    assert concat([x[:, :1], x[:, 1:]], axis=1).data == x.data

    assert raises(lambda: concat([a, c]), ValueError)
    assert raises(lambda: concat([a, tensor([1.0, 2.0])]), ValueError)
    assert raises(lambda: concat([a, b], axis=2), IndexError)
    assert raises(lambda: concat([]), ValueError)

    # the same tensor twice gets both grads
    concat([b, b]).sum().backward()
    assert b.grad == [2.0, 2.0]

    weights = tensor([0.3 * v - 2.0 for v in range(40)], shape=(2, 5, 4))
    gradcheck(lambda p, q: (concat([p, q], axis=1) * weights).sum(), x, y)


def test_log_softmax():
    logits = [[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0], [-5.0, -5.0, -5.0]]
    out = tensor(logits).log_softmax()
    for r, row in enumerate(logits):
        m = max(row)
        log_sum_exp = m + math.log(sum(math.exp(v - m) for v in row))
        for j, v in enumerate(row):
            assert close(out.data[r * 3 + j], v - log_sum_exp, abs_tol=1e-9), (r, j)
    # stays finite where log(softmax) would be log(0)
    assert out.data[5] == -2000.0

    x = tensor([[0.5, -0.3, 0.1], [2.0, 1.0, -1.0]])
    for a, b in zip(x.log_softmax().data, x.softmax().data):
        assert close(a, math.log(b))

    weights = tensor([[0.3, -1.0, 2.0], [1.5, 0.2, -0.7]])
    gradcheck(lambda t: (t.log_softmax() * weights).sum(), x)
    gradcheck(lambda t: (t.log_softmax() * tensor([0.5, 2.0, -1.0, 0.7])).sum(),
              tensor([0.1, -0.2, 0.3, 1.5]))


def test_nll_loss():
    logp = tensor([[-0.5, -1.0, -2.0], [-3.0, -0.1, -4.0]])
    assert close(logp.nll_loss([2, 1]).data[0], (2.0 + 0.1) / 2)
    assert logp.nll_loss(tensor([0, 0])).data == logp.nll_loss([0.0, 0.0]).data == [1.75]
    assert close(tensor([-0.2, -1.7]).nll_loss([1]).data[0], 1.7)

    # log_softmax + nll_loss is cross entropy with one-hot targets
    logits = [[0.3, -0.2, 0.5], [2.0, 0.1, -1.0], [1000.0, 0.0, -1000.0]]
    labels = [1, 0, 2]
    one_hot = [[1.0 if j == label else 0.0 for j in range(3)] for label in labels]
    loss = tensor(logits).log_softmax().nll_loss(labels).data[0]
    assert close(loss, ref_cross_entropy(logits, one_hot))
    assert close(loss, tensor(logits).cross_entropy_loss(tensor(one_hot)).data[0])

    # argmax output works as labels
    assert logp.nll_loss(logp.argmax(axis=1)).data == logp.nll_loss([0, 1]).data

    assert raises(lambda: logp.nll_loss([0]), ValueError)
    assert raises(lambda: logp.nll_loss([0, 3]), ValueError)
    assert raises(lambda: logp.nll_loss([0, -1]), ValueError)
    assert raises(lambda: logp.nll_loss([0, 1.5]), ValueError)

    x = tensor(logits[:2])
    gradcheck(lambda t: t.log_softmax().nll_loss([1, 0]), x)
    gradcheck(lambda t: (t * 2.0).nll_loss([2, 2]), x)


def test_binary_cross_entropy_with_logits():
    def ref(xs, ts):
        total = 0.0
        for x, t in zip(xs, ts):
            p = ref_sigmoid(x)
            total -= t * math.log(p) + (1 - t) * math.log(1 - p)
        return total / len(xs)

    xs, ts = [-2.0, -0.3, 0.0, 0.8, 3.0], [0.0, 1.0, 1.0, 0.0, 1.0]
    loss = tensor(xs).binary_cross_entropy_with_logits(tensor(ts)).data[0]
    assert close(loss, ref(xs, ts))
    assert close(tensor([0.0]).binary_cross_entropy_with_logits(tensor([1.0])).data[0], math.log(2))
    soft = tensor([[0.5, -1.0], [2.0, 0.1]]).binary_cross_entropy_with_logits(tensor([[0.2, 0.7], [0.9, 0.5]]))
    assert close(soft.data[0], ref([0.5, -1.0, 2.0, 0.1], [0.2, 0.7, 0.9, 0.5]))

    # confidently wrong: log(sigmoid(-1000)) would be log(0), here the loss is just 1000
    big = tensor([-1000.0, 1000.0]).binary_cross_entropy_with_logits(tensor([1.0, 0.0]))
    assert close(big.data[0], 1000.0)
    right = tensor([-1000.0, 1000.0]).binary_cross_entropy_with_logits(tensor([0.0, 1.0]))
    assert right.data[0] == 0.0

    assert raises(lambda: tensor([1.0, 2.0]).binary_cross_entropy_with_logits(tensor([1.0])), ValueError)

    targets = tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.3]])
    gradcheck(lambda t: t.binary_cross_entropy_with_logits(targets),
              tensor([[-2.0, -0.3, 0.4], [0.8, 3.0, -1.2]]))


def test_accuracy():
    scores = tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5], [0.2, 0.1, 0.9], [1.0, 1.0, 0.0]])
    assert scores.accuracy([1, 0, 2, 0]) == 1.0
    assert scores.accuracy([1, 0, 0, 1]) == 0.5
    assert scores.accuracy(tensor([0, 1, 0, 1])) == 0.0
    # same answer from logits and from probabilities
    assert scores.softmax().accuracy([1, 2, 2, 0]) == scores.accuracy([1, 2, 2, 0]) == 0.75
    assert tensor([0.3, 0.9]).accuracy([1]) == 1.0
    assert raises(lambda: scores.accuracy([1, 0]), ValueError)


def test_parameters_found_automatically():
    random.seed(6)
    layer = linear(3, 2)
    assert layer.parameters() == [layer.w, layer.b]

    mlp = MLP([3, 4, 1], dropout_p=0.5)
    assert mlp.parameters() == [mlp.layers[0].w, mlp.layers[0].b, mlp.layers[1].w, mlp.layers[1].b]
    assert dropout(0.5).parameters() == []

    # a custom module only stores its tensors and layers
    class scaled_linear(module):
        def __init__(self):
            super().__init__()
            self.scale = tensor([2.0])
            self.inner = linear(3, 2)

        def __call__(self, x):
            return self.inner(x) * self.scale

    custom = scaled_linear()
    assert custom.parameters() == [custom.scale, custom.inner.w, custom.inner.b]

    # a layer used twice is listed once
    shared = linear(2, 2)
    assert sequential(shared, shared).parameters() == [shared.w, shared.b]


def test_sequential():
    random.seed(7)
    first, second = linear(3, 4), linear(4, 2)
    drop = dropout(0.5)
    model = sequential(first, lambda x: x.relu(), drop, second)
    assert model.parameters() == [first.w, first.b, second.w, second.b]

    x = tensor([[0.5, -1.0, 2.0], [1.0, 0.3, -0.2]])
    model.eval()
    assert not drop.training
    assert model(x).data == second(first(x).relu()).data
    model.train()
    assert drop.training

    model.eval()
    y = tensor([[1.0, 0.0], [0.0, 1.0]])
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    model.train()
    opt = SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(30):
        loss = sequential(first, lambda x: x.relu(), second)(x).mse_loss(y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.data[0])
    assert losses[-1] < losses[0]


def test_sgd_momentum_and_weight_decay():
    # one weight, gradient fixed at 1: velocity goes 1, 1.9, 2.71
    w = tensor([0.0])
    opt = SGD([w], lr=0.1, momentum=0.9)
    expected, v = 0.0, 0.0
    for _ in range(3):
        w.grad = [1.0]
        opt.step()
        v = 0.9 * v + 1.0
        expected -= 0.1 * v
        assert close(w.data[0], expected)
    assert close(opt.velocity[0][0], 2.71)

    # with no gradient, weight decay alone shrinks the weight by (1 - lr * wd) each step
    w = tensor([2.0, -4.0])
    opt = SGD([w], lr=0.1, weight_decay=0.5)
    for _ in range(4):
        opt.zero_grad()
        opt.step()
    assert close(w.data[0], 2.0 * 0.95 ** 4) and close(w.data[1], -4.0 * 0.95 ** 4)

    # defaults are plain sgd
    w = tensor([1.0])
    w.grad = [3.0]
    SGD([w], lr=0.1).step()
    assert close(w.data[0], 0.7)

    assert raises(lambda: SGD([w], momentum=1.0), ValueError)
    assert raises(lambda: SGD([w], weight_decay=-0.1), ValueError)

    # momentum gets further than plain sgd on a long, shallow slope: f(w) = 0.01 * w^2
    finals = []
    for momentum in [0.0, 0.9]:
        w = tensor([10.0])
        opt = SGD([w], lr=0.5, momentum=momentum)
        for _ in range(50):
            opt.zero_grad()
            ((w * w) * 0.01).sum().backward()
            opt.step()
        finals.append(abs(w.data[0]))
    assert finals[1] < finals[0]


def test_adamw():
    # one step by hand: shrink 1.0 by (1 - 0.1 * 0.5), then the first Adam step moves
    # by lr * g / (|g| + eps), which is almost exactly lr
    w = tensor([1.0])
    w.grad = [2.0]
    AdamW([w], lr=0.1, weight_decay=0.5).step()
    assert close(w.data[0], 0.95 - 0.1 * 2.0 / (2.0 + 1e-8))

    # with no gradient, only the decay acts
    w = tensor([2.0, -4.0])
    opt = AdamW([w], lr=0.1, weight_decay=0.5)
    for _ in range(4):
        opt.zero_grad()
        opt.step()
    assert close(w.data[0], 2.0 * 0.95 ** 4) and close(w.data[1], -4.0 * 0.95 ** 4)

    # decoupled: the decay is the same whatever the gradient's size. with L2 in the
    # gradient, Adam would divide it away and a big gradient would hide the decay
    moved = []
    for g in [1.0, 1000.0]:
        a, b = tensor([1.0]), tensor([1.0])
        a.grad, b.grad = [g], [g]
        AdamW([a], lr=0.1, weight_decay=0.5).step()
        AdamW([b], lr=0.1, weight_decay=0.0).step()
        moved.append(b.data[0] - a.data[0])
    assert close(moved[0], moved[1]) and close(moved[0], 0.05)

    # weight_decay=0 is exactly Adam
    random.seed(8)
    x, y = rand_tensor(4, 3), rand_tensor(4, 1)
    finals = []
    for opt_class in [Adam, lambda params, lr: AdamW(params, lr, weight_decay=0.0)]:
        random.seed(9)
        model = MLP([3, 4, 1])
        opt = opt_class(model.parameters(), lr=0.05)
        for _ in range(10):
            loss = model(x).mse_loss(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        finals.append([v for p in model.parameters() for v in p.data])
    assert finals[0] == finals[1]

    assert raises(lambda: AdamW([w], weight_decay=-1.0), ValueError)


def test_clip_grad_norm():
    # gradients (3, 4) and (12,) make one vector of length 13
    a, b = tensor([1.0, 1.0]), tensor([1.0])
    a.grad, b.grad = [3.0, 4.0], [12.0]
    assert close(clip_grad_norm([a, b], max_norm=6.5), 13.0)
    assert close(a.grad[0], 1.5) and close(a.grad[1], 2.0) and close(b.grad[0], 6.0)
    new_norm = sum(g * g for g in a.grad + b.grad) ** 0.5
    assert close(new_norm, 6.5)

    # under the limit nothing changes
    a.grad = [0.3, 0.4]
    b.grad = [0.0]
    assert close(clip_grad_norm([a, b], max_norm=1.0), 0.5)
    assert a.grad == [0.3, 0.4] and b.grad == [0.0]

    # works on a model after backward, and the next step uses the clipped grads
    random.seed(10)
    model = MLP([3, 4, 1])
    x, y = rand_tensor(4, 3), tensor([[100.0], [-100.0], [100.0], [-100.0]])
    model(x).mse_loss(y).backward()
    before = clip_grad_norm(model.parameters(), max_norm=1.0)
    assert before > 1.0
    after = sum(g * g for p in model.parameters() for g in p.grad) ** 0.5
    assert close(after, 1.0)

    assert raises(lambda: clip_grad_norm([a], max_norm=0.0), ValueError)


def test_lr_schedules():
    # step: halve every 3 epochs
    opt = SGD([tensor([0.0])], lr=0.1)
    sched = StepLR(opt, step_size=3, gamma=0.5)
    lrs = [opt.lr]
    for _ in range(9):
        sched.step()
        lrs.append(opt.lr)
    for epoch, lr in enumerate(lrs):
        assert close(lr, 0.1 * 0.5 ** (epoch // 3))
    assert close(lrs[2], 0.1) and close(lrs[3], 0.05) and close(lrs[9], 0.0125)

    # cosine: compare every epoch against math.cos, then check it stays at eta_min
    opt = Adam([tensor([0.0])], lr=0.2)
    sched = CosineAnnealingLR(opt, T_max=100, eta_min=0.01)
    assert close(opt.lr, 0.2)
    for epoch in range(1, 101):
        sched.step()
        expected = 0.01 + (0.2 - 0.01) * (1 + math.cos(math.pi * epoch / 100)) / 2
        assert close(opt.lr, expected)
        if epoch == 50:
            assert close(opt.lr, (0.2 + 0.01) / 2)
    assert close(opt.lr, 0.01)
    for _ in range(5):
        sched.step()
        assert close(opt.lr, 0.01)

    # the optimizer steps with the scheduled lr
    w = tensor([0.0])
    opt = SGD([w], lr=0.1)
    sched = StepLR(opt, step_size=1, gamma=0.5)
    for _ in range(3):
        w.grad = [1.0]
        opt.step()
        sched.step()
    assert close(w.data[0], -(0.1 + 0.05 + 0.025))

    assert raises(lambda: StepLR(opt, step_size=0), ValueError)
    assert raises(lambda: StepLR(opt, step_size=2.5), ValueError)
    assert raises(lambda: StepLR(opt, step_size=2, gamma=0.0), ValueError)
    assert raises(lambda: CosineAnnealingLR(opt, T_max=0), ValueError)
    assert raises(lambda: CosineAnnealingLR(opt, T_max=10, eta_min=-0.1), ValueError)


def test_batches():
    # row i of x is [i, 10i], label i, so every pair can be checked after shuffling
    x = tensor([[float(i), 10.0 * i] for i in range(10)])
    labels = list(range(10))

    in_order = list(batches(x, labels, batch_size=4, shuffle=False))
    assert [len(yb) for _, yb in in_order] == [4, 4, 2]
    assert in_order[0][0].shape == (4, 2) and in_order[2][0].shape == (2, 2)
    assert in_order[1][0].data == [4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0]
    assert in_order[1][1] == [4, 5, 6, 7]

    random.seed(11)
    orders = []
    for _ in range(2):
        seen = []
        for xb, yb in batches(x, labels, batch_size=3):
            for r, label in enumerate(yb):
                assert xb.data[2 * r] == label and xb.data[2 * r + 1] == 10.0 * label
            seen += yb
        assert sorted(seen) == labels
        orders.append(seen)
    assert orders[0] != labels and orders[0] != orders[1]

    # y as a tensor, and 3-d rows keep their shape
    y = tensor([[float(i)] for i in range(10)])
    xb, yb = next(batches(x, y, batch_size=5, shuffle=False))
    assert yb.shape == (5, 1) and yb.data == [0.0, 1.0, 2.0, 3.0, 4.0]
    images = tensor([[[float(i)] * 2] * 2 for i in range(6)])
    xb, yb = next(batches(images, labels[:6], batch_size=4, shuffle=False))
    assert xb.shape == (4, 2, 2) and xb.data[4:8] == [1.0] * 4

    # batches are fresh tensors, outside the graph
    assert xb._prev == set()

    assert raises(lambda: next(batches(x, labels[:9], 4)), ValueError)
    assert raises(lambda: next(batches(x, labels, 0)), ValueError)
    assert raises(lambda: next(batches(x, labels, 2.0)), ValueError)


def test_save_and_load():
    random.seed(12)
    model = MLP([3, 4, 2])
    names = [name for name, _ in model.named_parameters()]
    assert names == ["layers.0.w", "layers.0.b", "layers.1.w", "layers.1.b"]
    assert [name for name, _ in sequential(linear(2, 2), model).named_parameters()][2:] == [
        "layers.1.layers.0.w", "layers.1.layers.0.b", "layers.1.layers.1.w", "layers.1.layers.1.b"]

    x = rand_tensor(5, 3)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "weights.json")
        model.save(path)
        with open(path) as f:
            state = json.load(f)
        assert list(state) == names and state["layers.0.w"]["shape"] == [3, 4]

        # a fresh model with other random weights gives the same numbers after loading
        other = MLP([3, 4, 2])
        assert other(x).data != model(x).data
        tensors_before = other.parameters()
        other.load(path)
        assert other(x).data == model(x).data
        assert all(a.data == b.data for a, b in zip(other.parameters(), model.parameters()))

        # loading copies into the same tensors, so an optimizer made earlier still updates them
        assert all(a is b for a, b in zip(other.parameters(), tensors_before))
        opt = SGD(tensors_before, lr=0.1)
        other(x).sum().backward()
        opt.step()
        assert other(x).data != model(x).data

        assert raises(lambda: MLP([3, 5, 2]).load(path), ValueError)
        assert raises(lambda: MLP([3, 4, 4, 2]).load(path), ValueError)
        assert raises(lambda: linear(3, 4).load(path), ValueError)

        # names come from where tensors are stored. sequential keeps its layers in
        # self.layers like MLP does, so the same two linears load, but a relu between
        # them moves the second one to layers.2
        stack = sequential(linear(3, 4), linear(4, 2))
        stack.load(path)
        assert [p.data for p in stack.parameters()] == [p.data for p in model.parameters()]
        assert raises(lambda: sequential(linear(3, 4), lambda h: h.relu(), linear(4, 2)).load(path), ValueError)


def test_layer_norm():
    random.seed(13)
    x = rand_tensor(3, 5) * 4.0 + 1.0
    out = x.layer_norm()
    for r in range(3):
        row = x.data[5 * r:5 * r + 5]
        mean = math.fsum(row) / 5
        var = math.fsum((v - mean) ** 2 for v in row) / 5
        for j in range(5):
            assert close(out.data[5 * r + j], (row[j] - mean) / math.sqrt(var + 1e-5))
        assert abs(math.fsum(out.data[5 * r:5 * r + 5])) < 1e-12

    # a constant row has nothing to normalize, and eps keeps it at 0 instead of 0 / 0
    assert tensor([[2.0, 2.0, 2.0]]).layer_norm().data == [0.0, 0.0, 0.0]

    # the sum of a normalized row is always 0, so weight the outputs to get real gradients
    r = rand_tensor(3, 5)
    gradcheck(lambda a: (a.layer_norm() * r).sum(), x)
    cube = tensor([[[random.uniform(-2, 2) for _ in range(4)] for _ in range(3)] for _ in range(2)])
    target = tensor([[[random.uniform(-1, 1) for _ in range(4)] for _ in range(3)] for _ in range(2)])
    gradcheck(lambda a: a.layer_norm(eps=0.1).mse_loss(target), cube)

    # the layer starts as a plain normalization, then gamma and beta get trained
    norm = layer_norm(5)
    assert [name for name, _ in norm.named_parameters()] == ["gamma", "beta"]
    assert norm(x).data == out.data
    norm.gamma.data = [random.uniform(0.5, 2) for _ in range(5)]
    norm.beta.data = [random.uniform(-1, 1) for _ in range(5)]
    y = rand_tensor(3, 5)
    gradcheck(lambda a, gamma, beta: norm(a).mse_loss(y), x, norm.gamma, norm.beta)

    model = sequential(linear(5, 8), layer_norm(8), lambda h: h.relu(), linear(8, 5))
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    assert raises(lambda: norm(rand_tensor(3, 4)), ValueError)


def test_batch_norm():
    random.seed(14)
    x = rand_tensor(6, 3) * 3.0 + 2.0
    norm = batch_norm(3)
    out = norm(x)

    # training mode: each column is normalized with its own batch mean and variance
    for j in range(3):
        col = x.data[j::3]
        mean = math.fsum(col) / 6
        var = math.fsum((v - mean) ** 2 for v in col) / 6
        for i in range(6):
            assert close(out.data[3 * i + j], (col[i] - mean) / math.sqrt(var + 1e-5))
        # running stats start at 0 and 1 and move 10% of the way to the batch values,
        # using the unbiased variance (divide by 5 instead of 6)
        assert close(norm.running_mean.data[j], 0.1 * mean)
        assert close(norm.running_var.data[j], 0.9 + 0.1 * var * 6 / 5)

    # after many steps on the same batch the running stats reach the batch values
    for _ in range(300):
        norm(x)
    for j in range(3):
        col = x.data[j::3]
        mean = math.fsum(col) / 6
        assert close(norm.running_mean.data[j], mean, rel=1e-9, abs_tol=1e-12)
        assert close(norm.running_var.data[j], math.fsum((v - mean) ** 2 for v in col) / 5)

    # eval mode uses the running stats, works on one row and leaves the stats alone
    norm.gamma.data = [2.0, 0.5, 1.0]
    norm.beta.data = [0.1, -0.2, 0.3]
    norm.eval()
    stats = norm.running_mean.data[:], norm.running_var.data[:]
    one = tensor([[1.0, -1.0, 0.5]])
    out = norm(one)
    for j in range(3):
        expected = (one.data[j] - stats[0][j]) / math.sqrt(stats[1][j] + 1e-5)
        assert close(out.data[j], expected * norm.gamma.data[j] + norm.beta.data[j])
    assert (norm.running_mean.data, norm.running_var.data) == stats

    # gradients, in both modes. in training, every input also moves the batch mean and var
    y = rand_tensor(6, 3)
    gradcheck(lambda a, gamma, beta: norm(a).mse_loss(y), x, norm.gamma, norm.beta)
    norm.train()
    gradcheck(lambda a, gamma, beta: norm(a).mse_loss(y), x, norm.gamma, norm.beta)
    model = sequential(linear(3, 4), batch_norm(4), lambda h: h.relu(), linear(4, 3))
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    # running stats are buffers: saved and loaded, but never given to the optimizer
    assert norm.parameters() == [norm.gamma, norm.beta]
    assert [name for name, _ in model.named_buffers()] == ["layers.1.running_mean", "layers.1.running_var"]
    stats = norm.running_mean.data[:], norm.running_var.data[:]
    opt = AdamW(norm.parameters(), lr=0.1, weight_decay=0.5)
    opt.step()
    assert (norm.running_mean.data, norm.running_var.data) == stats

    model.eval()
    copy = sequential(linear(3, 4), batch_norm(4), lambda h: h.relu(), linear(4, 3))
    copy.eval()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "weights.json")
        model.save(path)
        copy.load(path)
    assert copy.layers[1].running_var.data == model.layers[1].running_var.data
    assert copy(x).data == model(x).data

    assert raises(lambda: norm(tensor([[1.0, 2.0, 3.0]])), ValueError)
    assert raises(lambda: norm(rand_tensor(4, 2)), ValueError)
    assert raises(lambda: batch_norm(3, momentum=1.5), ValueError)


def test_embedding():
    random.seed(15)
    emb = embedding(5, 3)
    assert emb.weight.shape == (5, 3)
    assert [name for name, _ in emb.named_parameters()] == ["weight"]
    row = lambda i: emb.weight.data[3 * i:3 * i + 3]

    out = emb([[4, 0, 4], [2, 2, 1]])
    assert out.shape == (2, 3, 3)
    for n, i in enumerate([4, 0, 4, 2, 2, 1]):
        assert out.data[3 * n:3 * n + 3] == row(i)
    assert emb(3).shape == (3,) and emb(3).data == row(3)
    assert emb(tensor([1.0, 0.0])).data == row(1) + row(0)

    # a repeated id gets the grads of every place it was used; unused ids get none
    emb([1, 1, 2]).sum().backward()
    assert emb.weight.grad == [0.0] * 3 + [2.0] * 3 + [1.0] * 3 + [0.0] * 6

    ids = [[0, 3], [3, 3]]
    r = tensor([[[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)] for _ in range(2)])
    gradcheck(lambda w: (emb(ids) * r).sum(), emb.weight)

    # same values and grads as multiplying one-hot rows by the table
    flat = [0, 3, 3, 3]
    one_hot = tensor([[1.0 if c == i else 0.0 for c in range(5)] for i in flat])
    y = rand_tensor(4, 3)
    emb.zero_grad()
    emb(flat).mse_loss(y).backward()
    lookup_grad = emb.weight.grad[:]
    emb.zero_grad()
    product = one_hot @ emb.weight
    assert product.data == emb(flat).data
    product.mse_loss(y).backward()
    assert all(close(a, b) for a, b in zip(lookup_grad, emb.weight.grad))

    # init is standard normal
    random.seed(16)
    big = embedding(200, 50).weight.data
    mean = sum(big) / len(big)
    assert abs(mean) < 0.03 and abs(sum((v - mean) ** 2 for v in big) / len(big) - 1.0) < 0.05

    # learns: embed each of 4 ids in 2 dims, then a linear layer reads the id back out
    random.seed(17)
    model = sequential(embedding(4, 2), linear(2, 4))
    opt = Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        loss = model([0, 1, 2, 3]).log_softmax().nll_loss([0, 1, 2, 3])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert model([0, 1, 2, 3]).accuracy([0, 1, 2, 3]) == 1.0

    for bad in [-1, 5, 1.5, True]:
        assert raises(lambda: emb([0, bad]), ValueError)


def test_xavier_and_kaiming_init():
    random.seed(18)

    def variance(t):
        mean = math.fsum(t.data) / len(t.data)
        return math.fsum((v - mean) ** 2 for v in t.data) / len(t.data)

    # (fan_in, fan_out) = (300, 100), 30000 samples, so the variance lands within a few percent
    shape = (300, 100)
    blank = lambda: tensor([0.0] * 30000, shape=shape)
    cases = [
        (xavier_uniform(blank()), 2 / 400, (6 / 400) ** 0.5),
        (xavier_uniform(blank(), gain=2.0), 4 * 2 / 400, 2 * (6 / 400) ** 0.5),
        (xavier_normal(blank()), 2 / 400, None),
        (kaiming_uniform(blank()), 2 / 300, (6 / 300) ** 0.5),
        (kaiming_uniform(blank(), negative_slope=0.5), 2 / (1.25 * 300), (6 / (1.25 * 300)) ** 0.5),
        (kaiming_normal(blank()), 2 / 300, None),
    ]
    for w, expected_var, bound in cases:
        assert abs(variance(w) / expected_var - 1) < 0.05
        assert abs(math.fsum(w.data) / len(w.data)) < 0.1 * expected_var ** 0.5
        if bound is not None:
            assert max(abs(v) for v in w.data) <= bound
            assert max(abs(v) for v in w.data) > 0.99 * bound

    # fills in place, so a layer's weight can be re-initialized directly
    layer = linear(4, 3)
    assert kaiming_normal(layer.w) is layer.w and layer.parameters()[0] is layer.w

    # through 10 relu layers the signal keeps roughly its size with kaiming, but shrinks
    # about 6x per layer with linear's default (var(w) = 1 / (3 fan_in), then relu halves it).
    # narrow random layers make the kaiming size wander: 0.06 to 8.5 over seeds 0-99,
    # against at most 2e-7 with the default
    random.seed(19)
    x = tensor([[random.gauss(0.0, 1.0) for _ in range(32)] for _ in range(16)])
    sizes = []
    for init in [None, kaiming_normal]:
        h = x
        for _ in range(10):
            layer = linear(32, 32)
            if init:
                init(layer.w)
            h = layer(h).relu()
        sizes.append(math.fsum(v * v for v in h.data) / len(h.data))
    assert 0.01 < sizes[1] < 100
    assert sizes[0] < 1e-5

    for init in [xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal]:
        assert raises(lambda: init(tensor([1.0, 2.0])), ValueError)


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
