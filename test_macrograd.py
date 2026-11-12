import gc
import json
import math
import os
import random
import tempfile
import weakref

from engine import tensor, concat, gradcheck, no_grad, is_grad_enabled, stack, where
from nn import MLP, avg_pool2d, batch_norm, conv2d, dropout, embedding, flatten, gru_cell, layer_norm, linear, lstm_cell, max_pool2d, module, multi_head_attention, rms_norm, rnn_cell, sequential, transformer_block
from nn import xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, scaled_dot_product_attention, sample, sinusoidal_positions
from optim import SGD, Adagrad, RMSprop, Adam, AdamW, clip_grad_norm, StepLR, CosineAnnealingLR, LinearWarmup
from data import batches


def close(a, b, rel=1e-9, abs_tol=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def raises(fn, error):
    try:
        fn()
    except error:
        return True
    return False


def raises_with(fn, error, text):
    # like raises, but the message has to contain text, so an error from some op
    # further in doesn't count
    try:
        fn()
    except error as e:
        return text in str(e)
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


def test_silu():
    points = [-1000.0, -20.0, -3.0, -1.0, -0.2, 0.0, 0.5, 1.0, 3.0, 20.0, 1000.0]
    for x in points:
        assert close(tensor([x]).silu().data[0], x * ref_sigmoid(x)), x

    # grads against s * (1 + x * (1 - s)) worked out with math, and against finite differences
    a = tensor(points)
    weights = [0.5 * k - 2.0 for k in range(len(points))]
    (a.silu() * tensor(weights)).sum().backward()
    for x, w, g in zip(points, weights, a.grad):
        s = ref_sigmoid(x)
        assert close(g, w * s * (1 + x * (1 - s))), x
    assert a.grad[0] == 0.0 and a.grad[-1] == weights[-1]
    gradcheck(lambda a: (a.silu() * tensor(weights[1:-1])).sum(), tensor(points[1:-1]))
    # a is used twice here, so the grads from both uses have to add up
    gradcheck(lambda a: (a.silu() * a).sum(), tensor(points[1:-1]))

    # the dip: the lowest value is about -0.278, near x = -1.278, where the slope is 0
    dip = tensor([-1.2784645])
    assert close(dip.silu().data[0], -0.2784645, rel=1e-6)
    dip.silu().backward()
    assert abs(dip.grad[0]) < 1e-6

    random.seed(3)
    model = MLP([3, 5, 2], activation_fns=[lambda x: x.silu()])
    x, y = rand_tensor(4, 3), rand_tensor(4, 2)
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())


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


def test_cumsum():
    a = tensor([1.0, 2.0, 3.0, 4.0])
    c = a.cumsum(0)
    assert c.data == [1.0, 3.0, 6.0, 10.0] and c.shape == (4,)
    # a[j] is in every running total from position j on, so it's counted 4 - j times
    c.sum().backward()
    assert a.grad == [4.0, 3.0, 2.0, 1.0]

    m = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert m.cumsum(0).data == [1.0, 2.0, 3.0, 5.0, 7.0, 9.0] and m.cumsum(0).shape == (2, 3)
    assert m.cumsum(1).data == [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    assert m.cumsum(-1).data == m.cumsum(1).data

    random.seed(41)
    x = rand_shape(2, 3, 4)
    for axis in [0, 1, 2]:
        c = x.cumsum(axis)
        assert c.shape == x.shape
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    index = [i, j, k]
                    expected = math.fsum(x.get_element(*(index[:axis] + [t] + index[axis + 1:]))
                                         for t in range(index[axis] + 1))
                    assert close(c.get_element(*index), expected)
        # the last running total along the axis is the sum over it
        last = [slice(None)] * 3
        last[axis] = -1
        assert c[tuple(last)].data == x.sum(axis=axis).data

        # distinct weights, so a grad landing in the wrong position would show up
        weights = tensor([0.3 * v - 1.5 for v in range(24)], shape=x.shape)
        gradcheck(lambda t: (t.cumsum(axis) * weights).sum(), x)
    gradcheck(lambda t: (t.cumsum(1) * t).sum(), x)
    gradcheck(lambda t: t.cumsum(0).cumsum(2).tanh().sum(), x)

    assert raises(lambda: x.cumsum(3), IndexError)
    assert raises(lambda: x.cumsum(-4), IndexError)
    assert raises(lambda: tensor(3.0).cumsum(0), IndexError)


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


def test_min_and_argmin():
    a = tensor([[1.0, 7.0, 3.0], [-2.0, 9.0, -2.0]])
    assert a.min().data == [-2.0] and a.min().shape == ()
    assert a.argmin().data == [3]
    assert a.min(axis=1).data == [1.0, -2.0] and a.argmin(axis=1).data == [0, 0]
    assert a.min(axis=0).data == [-2.0, 7.0, -2.0] and a.argmin(axis=0).data == [1, 0, 1]
    assert a.min(axis=-1, keepdim=True).shape == (2, 1)
    assert a.argmin(axis=0, keepdim=True).shape == (1, 3)

    random.seed(38)
    x = tensor([random.uniform(-1, 1) for _ in range(24)], shape=(2, 3, 4))
    for axis in [0, 1, 2]:
        m, idx = x.min(axis=axis), x.argmin(axis=axis)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    out_index = [i, j, k]
                    del out_index[axis]
                    column = [x.get_element(*(out_index[:axis] + [t] + out_index[axis:]))
                              for t in range(x.shape[axis])]
                    assert m.get_element(*out_index) == min(column)
                    assert idx.get_element(*out_index) == column.index(min(column))

    # ties: the grad goes only to the first min, the one argmin points at
    t = tensor([5.0, 2.0, 2.0])
    t.min().backward()
    assert t.grad == [0.0, 1.0, 0.0]

    assert raises(lambda: a.min(axis=2), IndexError)
    assert raises(lambda: tensor([], shape=(0,)).min(), ValueError)

    for axis in [None, 0, 1, 2]:
        out = x.min(axis=axis)
        weights = tensor([0.5 * v - 2.0 for v in range(len(out.data))], shape=out.shape)
        gradcheck(lambda t: (t.min(axis=axis) * weights).sum(), x)


def test_clamp():
    points = [-3.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 7.0]
    x = tensor(points)
    for lo, hi in [(-0.5, 1.0), (0.0, None), (None, 0.5), (2.0, 2.0)]:
        out = x.clamp(min=lo, max=hi)
        expected = [min(max(v, lo if lo is not None else -math.inf), hi if hi is not None else math.inf) for v in points]
        assert out.data == expected, (lo, hi)

        # slope 1 where the value is inside the range, bounds included, and 0 where it was cut off
        weights = [0.5 * k - 1.0 for k in range(len(points))]
        x.grad = [0.0] * len(points)
        (out * tensor(weights)).sum().backward()
        for v, w, g in zip(points, weights, x.grad):
            inside = (lo is None or v >= lo) and (hi is None or v <= hi)
            assert g == (w if inside else 0.0), (lo, hi, v)

    # clamp(min=0) is relu, and it can stand in for relu6, min(max(x, 0), 6)
    assert x.clamp(min=0.0).data == x.relu().data
    assert x.clamp(0.0, 6.0).data == [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 2.0, 6.0]
    assert x.clamp(max=1).data == [-3.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1, 1]

    # finite differences need room on both sides, so keep the points away from the bounds
    random.seed(39)
    a = tensor([[random.uniform(-2, 2) for _ in range(5)] for _ in range(3)])
    lo, hi = -0.7, 0.9
    assert all(abs(v - lo) > 1e-3 and abs(v - hi) > 1e-3 for v in a.data)
    gradcheck(lambda t: (t.clamp(lo, hi) * t).sum(), a)

    assert raises(lambda: x.clamp(), ValueError)
    assert raises(lambda: x.clamp(1.0, 0.0), ValueError)


def test_where():
    cond = [[True, False, True], [False, False, True]]
    a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    out = where(cond, a, b)
    assert out.shape == (2, 3) and out.data == [1.0, -2.0, 3.0, -4.0, -5.0, 6.0]
    # a condition tensor counts nonzero as true
    assert where(tensor([[1, 0, 2], [0, 0, -1]]), a, b).data == out.data

    # each grad goes to the side that was picked
    weights = tensor([[0.5, -1.0, 2.0], [3.0, -0.25, 1.5]])
    (out * weights).sum().backward()
    assert a.grad == [0.5, 0.0, 2.0, 0.0, 0.0, 1.5]
    assert b.grad == [0.0, -1.0, 0.0, 3.0, -0.25, 0.0]

    # broadcasting: a (2, 3) condition, a (3,) row and a number
    row = tensor([10.0, 20.0, 30.0])
    out = where(cond, row, 0.5)
    assert out.data == [10.0, 0.5, 30.0, 0.5, 0.5, 30.0]
    (out * weights).sum().backward()
    assert row.grad == [0.5, 0.0, 2.0 + 1.5]
    # a (2, 1) condition picks whole rows, and b alone can set the shape too
    assert where([[True], [False]], a, b).data == [1.0, 2.0, 3.0, -4.0, -5.0, -6.0]
    assert where([True, False, True], 0.0, b).data == [0.0, -2.0, 0.0, 0.0, -5.0, 0.0]

    random.seed(39)
    x, y = rand_tensor(2, 3), rand_tensor(1, 3)
    gradcheck(lambda p, q: (where(cond, p * q, p.tanh()) * weights).sum(), x, y)
    # p is used twice, so its grads from where and from * have to add up
    z = rand_tensor(2, 3)
    gradcheck(lambda p, q: (where(cond, p, q) * p).sum(), x, z)

    # the condition only chooses: it gets no grad and isn't in the graph
    c = tensor([1.0, 0.0])
    picked = where(c, tensor([1.0, 2.0]), tensor([3.0, 4.0]))
    assert c not in picked._prev
    picked.sum().backward()
    assert c.grad == [0.0, 0.0]

    # masking: -inf scores get probability 0, and no grad and no nan flow to them
    scores = tensor([[0.3, 1.2, -0.7, 2.0]])
    mask = [[True, True, False, True]]
    probs = where(mask, scores, float("-inf")).softmax()
    kept = [math.exp(v) for v in [0.3, 1.2, 2.0]]
    assert all(close(p, e) for p, e in zip([probs.data[0], probs.data[1], probs.data[3]], [k / sum(kept) for k in kept]))
    assert probs.data[2] == 0.0
    (probs * tensor([[1.0, -2.0, 5.0, 0.5]])).sum().backward()
    assert scores.grad[2] == 0.0 and all(g == g for g in scores.grad)

    assert raises(lambda: where([True, False], a, b), ValueError)
    assert raises(lambda: where(cond, a, tensor([1.0, 2.0])), ValueError)


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


def test_stack():
    a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    assert stack([a, b]).shape == (2, 2, 3)
    assert stack([a, b]).data == a.data + b.data
    assert stack([a, b], axis=1).shape == (2, 2, 3)
    assert stack([a, b], axis=1).data == [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
    assert stack([a, b], axis=-1).shape == (2, 3, 2)
    assert stack([a, b], axis=2).data == [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0]

    # every axis, checked value by value: out[..., n, ...] is the n-th tensor
    random.seed(40)
    ts = [rand_tensor(2, 3) for _ in range(4)]
    for axis in [0, 1, 2, -1, -2, -3]:
        out = stack(ts, axis)
        k = axis % 3
        assert out.shape == (2, 3)[:k] + (4,) + (2, 3)[k:]
        for n, t in enumerate(ts):
            for i in range(2):
                for j in range(3):
                    index = [i, j]
                    index.insert(k, n)
                    assert out.get_element(*index) == t.get_element(i, j)

    # numbers stack into a vector, and indexing undoes stacking
    assert stack([tensor(1.5), tensor(-2.0), tensor(0.0)]).data == [1.5, -2.0, 0.0]
    assert stack([tensor(1.5), tensor(-2.0)]).shape == (2,)
    assert stack([a, b])[1].data == b.data and stack([a, b], axis=1)[:, 0].data == a.data

    # hidden states from each step of an rnn, stacked along time: (batch, time, hidden)
    cell = rnn_cell(3, 4)
    xs, h, hs = [rand_tensor(2, 3) for _ in range(5)], None, []
    for x in xs:
        h = cell(x, h)
        hs.append(h)
    assert stack(hs, axis=1).shape == (2, 5, 4)
    assert stack(hs, axis=1)[:, 4].data == h.data

    weights = tensor([0.3 * v - 2.0 for v in range(24)], shape=(2, 4, 3))
    gradcheck(lambda *t: (stack(t, axis=1) * weights).sum(), *ts)
    # the same tensor twice gets both grads
    c = tensor([1.0, 2.0])
    (stack([c, c]) * tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    assert c.grad == [4.0, 6.0]

    # concat would refuse these too, but the message should say what stack needs
    try:
        stack([a, tensor([1.0, 2.0, 3.0])])
        assert False, "stack accepted different shapes"
    except ValueError as e:
        assert "same shape" in str(e)
    assert raises(lambda: stack([]), ValueError)
    assert raises(lambda: stack([a, b], axis=3), IndexError)
    assert raises(lambda: stack([a, b], axis=-4), IndexError)


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


def test_nll_loss_label_smoothing():
    logp = tensor([[-0.5, -1.0, -2.0], [-3.0, -0.1, -4.0]])
    assert logp.nll_loss([2, 1], label_smoothing=0.0).data == logp.nll_loss([2, 1]).data
    # by hand: the target is 0.9 + 0.1/3 on the label and 0.1/3 on the others
    e = 0.1
    expected = -math.fsum([(1 - e) * -2.0 + e / 3 * -3.5, (1 - e) * -0.1 + e / 3 * -7.1]) / 2
    assert close(logp.nll_loss([2, 1], label_smoothing=e).data[0], expected)

    # after log_softmax it's cross entropy against the smoothed target
    logits = [[0.3, -0.2, 0.5, 1.0], [2.0, 0.1, -1.0, 0.0], [1000.0, 0.0, -1000.0, 3.0]]
    labels = [1, 0, 2]
    for e in [0.1, 0.25, 1.0]:
        smoothed = [[(1 - e) * (j == label) + e / 4 for j in range(4)] for label in labels]
        loss = tensor(logits).log_softmax().nll_loss(labels, label_smoothing=e).data[0]
        assert close(loss, ref_cross_entropy(logits, smoothed))
        assert close(loss, tensor(logits).cross_entropy_loss(tensor(smoothed)).data[0])
    # with e = 1 the target is uniform and the labels don't matter
    assert logp.nll_loss([0, 0], label_smoothing=1.0).data == logp.nll_loss([2, 1], label_smoothing=1.0).data

    # the lowest loss is when the model predicts the smoothed target itself: the grad is 0
    # there, and the loss is the target's entropy. one-hot targets would push on forever
    e, label = 0.2, 3
    target = [(1 - e) * (j == label) + e / 5 for j in range(5)]
    x = tensor([[math.log(q) for q in target]])
    loss = x.log_softmax().nll_loss([label], label_smoothing=e)
    loss.backward()
    assert close(loss.data[0], -math.fsum(q * math.log(q) for q in target))
    assert all(abs(g) < 1e-12 for g in x.grad)
    x.grad = [0.0] * 5
    x.log_softmax().nll_loss([label]).backward()
    assert close(x.grad[label], target[label] - 1)

    # a class with probability 0 only matters if the target gives it some weight
    logp = tensor([[0.0, float("-inf")]])
    assert logp.nll_loss([0]).data == [0.0]
    assert logp.nll_loss([0], label_smoothing=0.1).data == [float("inf")]
    assert logp.nll_loss([1], label_smoothing=1.0).data == [float("inf")]

    x = tensor(logits[:2])
    for e in [0.1, 0.6, 1.0]:
        gradcheck(lambda t: t.log_softmax().nll_loss([1, 3], label_smoothing=e), x)
        gradcheck(lambda t: (t * 2.0).nll_loss([2, 2], label_smoothing=e), x)

    for bad in [-0.1, 1.5, True, float("nan")]:
        assert raises_with(lambda: logp.nll_loss([0], label_smoothing=bad), ValueError, "label_smoothing")


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


def test_adagrad():
    # steady gradient of 1: after t steps s = t, so step t moves lr / sqrt(t)
    w = tensor([0.0])
    opt = Adagrad([w], lr=0.1)
    expected = 0.0
    for t in range(1, 6):
        w.grad = [1.0]
        opt.step()
        expected -= 0.1 / t ** 0.5
        assert close(w.data[0], expected)
    assert opt.sum_sq == [[5.0]]

    # every weight gets its own scale: gradients of 0.01 and 100 give the same steps
    w = tensor([0.0, 0.0])
    opt = Adagrad([w], lr=0.1)
    for _ in range(3):
        w.grad = [0.01, 100.0]
        opt.step()
    # (equal up to eps, which is 1e-8 of sqrt(s) for the small gradient)
    assert close(w.data[0], w.data[1], rel=1e-6) and close(w.data[1], -0.1 * (1 + 2 ** -0.5 + 3 ** -0.5))

    # a weight that gets its first gradient late takes a full lr step, while one that got a
    # gradient every step is down to lr / sqrt(10). a zero gradient doesn't move a weight
    w = tensor([0.0, 0.0])
    opt = Adagrad([w], lr=0.1)
    for t in range(10):
        before = w.data[:]
        w.grad = [1.0, 1.0 if t == 9 else 0.0]
        opt.step()
        if t < 9:
            assert w.data[1] == 0.0
    assert close(before[0] - w.data[0], 0.1 / 10 ** 0.5) and close(before[1] - w.data[1], 0.1)

    # eps is added after the square root, so a gradient as small as eps moves only lr / 2
    w = tensor([0.0])
    w.grad = [1e-10]
    Adagrad([w], lr=0.1).step()
    assert close(w.data[0], -0.05)

    assert raises(lambda: Adagrad([w], eps=0.0), ValueError)


def test_rmsprop():
    # steady gradient g: after t steps v = (1 - alpha^t) * g^2. v starts at 0, so the first
    # step is lr / sqrt(1 - alpha) = 10 * lr, and later steps come down toward lr
    w = tensor([0.0])
    opt = RMSprop([w], lr=0.01)
    expected = 0.0
    for t in range(1, 6):
        w.grad = [2.0]
        opt.step()
        expected -= 0.01 * 2.0 / ((1 - 0.99 ** t) ** 0.5 * 2.0 + 1e-8)
        assert close(w.data[0], expected)
        if t == 1:
            assert close(w.data[0], -0.1, rel=1e-6)
    assert close(opt.v[0][0], (1 - 0.99 ** 5) * 4.0)

    # every weight gets its own scale: gradients of 0.01 and 100 give the same steps
    w = tensor([0.0, 0.0])
    opt = RMSprop([w], lr=0.01, alpha=0.9)
    for _ in range(3):
        w.grad = [0.01, 100.0]
        opt.step()
    assert close(w.data[0], w.data[1], rel=1e-4)

    # after gradients of 10 for 100 steps and then 1 for 1000, RMSprop has forgotten the big
    # ones and steps about lr again. Adagrad still divides by sqrt(100 * 10^2 + 1000)
    a, b = tensor([0.0]), tensor([0.0])
    adagrad, rmsprop = Adagrad([a], lr=0.01), RMSprop([b], lr=0.01)
    for t in range(1100):
        before = a.data[0], b.data[0]
        a.grad = b.grad = [10.0 if t < 100 else 1.0]
        adagrad.step()
        rmsprop.step()
    assert close(before[0] - a.data[0], 0.01 / 11000 ** 0.5)
    assert abs((before[1] - b.data[0]) - 0.01) < 1e-4

    # both train a small network
    random.seed(10)
    x, y = rand_tensor(8, 3), rand_tensor(8, 1)
    for opt_class, lr in [(Adagrad, 0.1), (RMSprop, 0.01)]:
        random.seed(11)
        model = MLP([3, 8, 1])
        opt = opt_class(model.parameters(), lr=lr)
        first = model(x).mse_loss(y).data[0]
        for _ in range(100):
            loss = model(x).mse_loss(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert model(x).mse_loss(y).data[0] < 0.05 * first

    assert raises(lambda: RMSprop([w], alpha=1.0), ValueError)
    assert raises(lambda: RMSprop([w], alpha=-0.1), ValueError)
    assert raises(lambda: RMSprop([w], eps=0.0), ValueError)


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


def test_linear_warmup():
    def lrs_over(opt, sched, epochs):
        lrs = [opt.lr]
        for _ in range(epochs):
            sched.step()
            lrs.append(opt.lr)
        return lrs

    # 3 warmup steps from 0.4: equal steps of 0.1 up to 0.4, then it stays there.
    # the small lr is set as soon as the warmup is made, so epoch 0 uses it
    opt = SGD([tensor([0.0])], lr=0.4)
    sched = LinearWarmup(opt, 3)
    assert close(opt.lr, 0.1) and sched.base_lr == 0.4
    for got, want in zip(lrs_over(opt, sched, 6), [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]):
        assert close(got, want)

    # then cosine, starting from its own epoch 0 once the warmup is done
    opt = Adam([tensor([0.0])], lr=1.0)
    sched = LinearWarmup(opt, 2, after=CosineAnnealingLR(opt, T_max=4, eta_min=0.1))
    cosine = [0.1 + 0.9 * (1 + math.cos(math.pi * e / 4)) / 2 for e in range(5)]
    for got, want in zip(lrs_over(opt, sched, 8), [1 / 3, 2 / 3] + cosine + [0.1, 0.1]):
        assert close(got, want)

    # and StepLR the same way
    opt = SGD([tensor([0.0])], lr=0.8)
    sched = LinearWarmup(opt, 1, after=StepLR(opt, step_size=2, gamma=0.5))
    for got, want in zip(lrs_over(opt, sched, 5), [0.4, 0.8, 0.8, 0.4, 0.4, 0.2]):
        assert close(got, want)

    # the optimizer steps with the warmed-up lr from the first step on
    w = tensor([0.0])
    opt = SGD([w], lr=0.3)
    sched = LinearWarmup(opt, 2)
    for _ in range(4):
        w.grad = [1.0]
        opt.step()
        sched.step()
    assert close(w.data[0], -(0.1 + 0.2 + 0.3 + 0.3))

    # RMSprop's first step is 10 * lr, because its average of squared gradients starts at
    # 0. with 9 warmup steps the first step is lr instead
    w = tensor([0.0])
    opt = RMSprop([w], lr=0.01)
    LinearWarmup(opt, 9)
    w.grad = [5.0]
    opt.step()
    assert close(w.data[0], -0.01, rel=1e-6)

    other = SGD([tensor([0.0])], lr=0.1)
    assert raises(lambda: LinearWarmup(opt, 0), ValueError)
    assert raises(lambda: LinearWarmup(opt, 2.5), ValueError)
    assert raises(lambda: LinearWarmup(opt, True), ValueError)
    assert raises(lambda: LinearWarmup(opt, 3, after=StepLR(other, step_size=2)), ValueError)
    assert raises(lambda: LinearWarmup(opt, 3, after=lambda epoch: 0.1), ValueError)


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


def test_rms_norm():
    random.seed(37)
    x = rand_tensor(3, 5) * 4.0 + 1.0
    out = x.rms_norm()
    for r in range(3):
        row = x.data[5 * r:5 * r + 5]
        ms = math.fsum(v * v for v in row) / 5
        for j in range(5):
            assert close(out.data[5 * r + j], row[j] / math.sqrt(ms + 1e-5))
        # the squares average to ms / (ms + eps), just under 1
        assert close(math.fsum(v * v for v in out.data[5 * r:5 * r + 5]) / 5, ms / (ms + 1e-5))

    # unlike layer_norm it doesn't subtract the mean: a constant row stays (almost) 1s
    # instead of becoming 0s, and an all-zero row stays 0 thanks to eps
    assert all(close(v, 2.0 / math.sqrt(4.0 + 1e-5)) for v in tensor([[2.0, 2.0, 2.0]]).rms_norm().data)
    assert tensor([[0.0, 0.0, 0.0]]).rms_norm().data == [0.0, 0.0, 0.0]
    # scaling a row by c only changes the output through eps: it's the same as using eps / c^2
    big, small_eps = (x * 1000.0).rms_norm(), x.rms_norm(eps=1e-5 / 1000.0 ** 2)
    assert all(close(a, b) for a, b in zip(big.data, small_eps.data))
    assert all(abs(a - b) < 1e-5 for a, b in zip(big.data, out.data))

    # weight the outputs so the grads aren't hidden by symmetry
    w = rand_tensor(3, 5)
    gradcheck(lambda a: (a.rms_norm() * w).sum(), x)
    cube = tensor([[[random.uniform(-2, 2) for _ in range(4)] for _ in range(3)] for _ in range(2)])
    target = tensor([[[random.uniform(-1, 1) for _ in range(4)] for _ in range(3)] for _ in range(2)])
    gradcheck(lambda a: a.rms_norm(eps=0.1).mse_loss(target), cube)
    gradcheck(lambda a: (a.rms_norm() * a).sum(), x)

    # since scaling a row doesn't change the output, the grad of each row points across
    # the row, not along it: sum(x * dL/dx) is about 0 for every row
    x.grad = [0.0] * 15
    (x.rms_norm() * w).sum().backward()
    for r in range(3):
        assert abs(math.fsum(a * b for a, b in zip(x.data[5 * r:5 * r + 5], x.grad[5 * r:5 * r + 5]))) < 1e-5

    # the layer starts as a plain normalization, then gamma gets trained
    norm = rms_norm(5)
    assert [name for name, _ in norm.named_parameters()] == ["gamma"]
    assert norm(x).data == out.data
    norm.gamma.data = [random.uniform(0.5, 2) for _ in range(5)]
    y = rand_tensor(3, 5)
    for r in range(3):
        row = x.data[5 * r:5 * r + 5]
        rms = math.sqrt(math.fsum(v * v for v in row) / 5 + 1e-5)
        for j in range(5):
            assert close(norm(x).data[5 * r + j], row[j] / rms * norm.gamma.data[j])
    gradcheck(lambda a, gamma: norm(a).mse_loss(y), x, norm.gamma)

    model = sequential(linear(5, 8), rms_norm(8), lambda h: h.silu(), linear(8, 5))
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    assert raises(lambda: norm(rand_tensor(3, 4)), ValueError)
    assert repr(norm) == "RMSNorm(5, eps=1e-05)"
    loose = rms_norm(3, eps=0.5)
    assert all(close(v, 2.0 / math.sqrt(4.0 + 0.5)) for v in loose(tensor([[2.0, 2.0, 2.0]])).data)


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


def test_sinusoidal_positions():
    # against math.sin and math.cos, including positions far past 2 pi
    for length, dim, base in [(20, 8, 10000.0), (2000, 4, 10000.0), (5, 6, 100.0)]:
        pe = sinusoidal_positions(length, dim, base)
        assert pe.shape == (length, dim)
        for p in range(length):
            for i in range(dim // 2):
                w = 1 / base ** (2 * i / dim)
                assert close(pe.get_element(p, 2 * i), math.sin(p * w), abs_tol=1e-12)
                assert close(pe.get_element(p, 2 * i + 1), math.cos(p * w), abs_tol=1e-12)
    pe = sinusoidal_positions(2, 4)
    assert all(close(a, b, abs_tol=1e-15) for a, b in zip(pe.data, [0.0, 1.0, 0.0, 1.0,
                                                                 math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)]))

    # moving k positions on turns each pair by the same angle k * w, wherever it starts
    pe = sinusoidal_positions(30, 8)
    k = 7
    for i in range(4):
        w = 1 / 10000 ** (2 * i / 8)
        for p in range(30 - k):
            sin_p, cos_p = pe.get_element(p, 2 * i), pe.get_element(p, 2 * i + 1)
            assert close(pe.get_element(p + k, 2 * i), sin_p * math.cos(k * w) + cos_p * math.sin(k * w), abs_tol=1e-12)
            assert close(pe.get_element(p + k, 2 * i + 1), cos_p * math.cos(k * w) - sin_p * math.sin(k * w), abs_tol=1e-12)
    # so how alike two rows are depends only on how far apart they are, and near rows are more alike
    def dot(a, b):
        return math.fsum(x * y for x, y in zip(pe[a].data, pe[b].data))
    assert close(dot(3, 5), dot(20, 22), abs_tol=1e-12) and close(dot(10, 10), 4.0)
    assert dot(10, 10) > dot(10, 11) > dot(10, 13)

    # added to token vectors it gives the same token a different vector at each position
    random.seed(24)
    tok = embedding(10, 8)
    x = tok(tensor([[3, 3, 3], [1, 2, 3]])) + sinusoidal_positions(3, 8)
    assert x.shape == (2, 3, 8) and x[0, 0].data != x[0, 1].data
    assert all(close(a, b) for a, b in zip(x[1, 2].data, x[0, 2].data))

    assert raises_with(lambda: sinusoidal_positions(0, 4), ValueError, "length")
    assert raises_with(lambda: sinusoidal_positions(True, 4), ValueError, "length")
    assert raises_with(lambda: sinusoidal_positions(3.0, 4), ValueError, "length")
    assert raises_with(lambda: sinusoidal_positions(3, 0), ValueError, "dim")
    assert raises_with(lambda: sinusoidal_positions(3, 5), ValueError, "even")
    assert raises_with(lambda: sinusoidal_positions(3, 4, base=1.0), ValueError, "base")


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


def rand_shape(*shape, low=-1.0, high=1.0):
    size = 1
    for d in shape:
        size *= d
    return tensor([random.uniform(low, high) for _ in range(size)], shape=shape)


def ref_conv2d(x, w, stride, padding):
    # straight from the definition, reading 0 outside the image
    N, C, H, W = x.shape
    O, _, KH, KW = w.shape
    def pixel(n, c, y, z):
        return x.get_element(n, c, y, z) if 0 <= y < H and 0 <= z < W else 0.0
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    return [math.fsum(pixel(n, c, i * stride + a - padding, j * stride + b - padding) * w.get_element(o, c, a, b)
                      for c in range(C) for a in range(KH) for b in range(KW))
            for n in range(N) for o in range(O) for i in range(OH) for j in range(OW)]


def test_conv2d():
    # a 2x2 filter of ones adds up each 2x2 patch
    x = tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    ones = tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    assert x.conv2d(ones).data == [12.0, 16.0, 24.0, 28.0]
    # with a zero border the corners only see one pixel; stride 2 skips every other stop
    padded = x.conv2d(ones, padding=1)
    assert padded.shape == (1, 1, 4, 4) and padded.data[0] == 1.0 and padded.data[15] == 9.0
    assert x.conv2d(ones, stride=2, padding=1).data == [1.0, 5.0, 11.0, 28.0]

    random.seed(20)
    for N, C, H, W, O, K, stride, padding in [(2, 2, 5, 4, 3, 3, 1, 1), (1, 3, 6, 6, 2, 2, 2, 0),
                                              (2, 1, 5, 7, 2, 3, 2, 2), (1, 2, 4, 4, 1, 4, 1, 0)]:
        x, w = rand_shape(N, C, H, W), rand_shape(O, C, K, K)
        out = x.conv2d(w, stride, padding)
        OH, OW = (H + 2 * padding - K) // stride + 1, (W + 2 * padding - K) // stride + 1
        assert out.shape == (N, O, OH, OW)
        assert all(close(a, b) for a, b in zip(out.data, ref_conv2d(x, w, stride, padding)))

    # a filter as big as the image is one dot product per filter: a matmul on the flat image
    x, w = rand_shape(3, 2, 4, 4), rand_shape(5, 2, 4, 4)
    as_matmul = x.reshape(3, 32) @ w.reshape(5, 32).transpose()
    assert all(close(a, b) for a, b in zip(x.conv2d(w).data, as_matmul.data))
    # a 1x1 filter mixes the channels at each pixel, like a linear layer on every pixel
    w = rand_shape(4, 2, 1, 1)
    # (N, C, H, W) -> (N, W, H, C), one pixel per row, matmul, then back to (N, O, H, W)
    pixels = x.transpose(1, 3).reshape(48, 2)
    per_pixel = (pixels @ w.reshape(4, 2).transpose()).reshape(3, 4, 4, 4).transpose(1, 3)
    assert all(close(a, b) for a, b in zip(x.conv2d(w).data, per_pixel.data))

    x, w = rand_shape(2, 2, 4, 5), rand_shape(3, 2, 3, 3)
    for stride, padding in [(1, 0), (1, 1), (2, 1)]:
        r = rand_shape(*x.conv2d(w, stride, padding).shape)
        gradcheck(lambda a, f: (a.conv2d(f, stride, padding) * r).sum(), x, w)

    # the layer: one bias per filter, added to every pixel of that filter's output
    layer = conv2d(2, 3, kernel_size=3, padding=1)
    assert [name for name, _ in layer.named_parameters()] == ["w", "b"]
    assert layer.w.shape == (3, 2, 3, 3) and max(abs(v) for v in layer.w.data) <= 18 ** -0.5
    layer.b.data = [1.0, -2.0, 0.5]
    out, plain = layer(x), x.conv2d(layer.w, padding=1)
    for k in range(len(out.data)):
        assert close(out.data[k], plain.data[k] + layer.b.data[(k // 20) % 3])

    model = sequential(conv2d(2, 3, 3, padding=1), lambda h: h.relu(), conv2d(3, 2, 2, stride=2))
    y = rand_shape(2, 2, 2, 2)
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    # init works on conv weights, with fan-in C * K * K = 144 and fan-out O * K * K = 900
    w = tensor([0.0] * 14400, shape=(100, 16, 3, 3))
    for init, expected in [(kaiming_normal, 2 / 144), (xavier_uniform, 2 / (144 + 900))]:
        init(w)
        mean = math.fsum(w.data) / len(w.data)
        assert abs(math.fsum((v - mean) ** 2 for v in w.data) / len(w.data) / expected - 1) < 0.05

    image, filters = rand_shape(1, 2, 3, 3), rand_shape(1, 2, 2, 2)
    assert raises(lambda: image.conv2d(rand_shape(1, 3, 2, 2)), ValueError)
    assert raises(lambda: image.conv2d(rand_shape(1, 2, 4, 4)), ValueError)
    assert raises(lambda: image.conv2d(filters, stride=0), ValueError)
    assert raises(lambda: image.conv2d(filters, padding=-1), ValueError)
    assert raises(lambda: rand_shape(2, 3, 3).conv2d(filters), ValueError)


def test_max_pool2d_and_flatten():
    x = tensor([[[[1.0, 3.0, 2.0, 0.0],
                  [4.0, 2.0, 1.0, 5.0],
                  [0.0, 1.0, 9.0, 8.0],
                  [7.0, 6.0, 3.0, 2.0]]]])
    out = x.max_pool2d(2)
    assert out.shape == (1, 1, 2, 2) and out.data == [4.0, 5.0, 7.0, 9.0]

    # each grad goes to the winner of its window only
    (out * tensor([[[[1.0, 2.0], [3.0, 4.0]]]])).sum().backward()
    assert x.grad == [0.0, 0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0, 2.0,
                      0.0, 0.0, 4.0, 0.0,
                      3.0, 0.0, 0.0, 0.0]

    # overlapping windows (stride 1): the 9 wins all four windows and gets all four grads
    x = tensor([[[[1.0, 2.0, 0.0], [3.0, 9.0, 4.0], [0.0, 5.0, 1.0]]]])
    out = x.max_pool2d(2, stride=1)
    assert out.data == [9.0] * 4
    out.sum().backward()
    assert x.grad[4] == 4.0 and sum(x.grad) == 4.0

    # on a tie the first value wins
    x = tensor([[[[2.0, 2.0], [2.0, 2.0]]]])
    x.max_pool2d(2).sum().backward()
    assert x.grad == [1.0, 0.0, 0.0, 0.0]

    random.seed(21)
    for N, C, H, W, K, stride in [(2, 3, 5, 6, 2, None), (1, 2, 7, 7, 3, 2), (2, 1, 4, 5, 2, 1)]:
        x = rand_shape(N, C, H, W)
        S = stride or K
        out = x.max_pool2d(K, stride)
        OH, OW = (H - K) // S + 1, (W - K) // S + 1
        assert out.shape == (N, C, OH, OW)
        expected = [max(x.get_element(n, c, i * S + a, j * S + b) for a in range(K) for b in range(K))
                    for n in range(N) for c in range(C) for i in range(OH) for j in range(OW)]
        assert out.data == expected
        r = rand_shape(N, C, OH, OW)
        gradcheck(lambda a: (a.max_pool2d(K, stride) * r).sum(), x)

    # flatten is a reshape that keeps the order of the values
    x = rand_shape(2, 3, 4, 5)
    assert x.flatten().shape == (120,) and x.flatten().data == x.data
    assert x.flatten(1).shape == (2, 60) and x.flatten(2).shape == (2, 3, 20)
    assert x.flatten(-1).shape == (2, 3, 4, 5) and tensor(3.0).flatten().shape == (1,)
    r = rand_shape(2, 60)
    gradcheck(lambda a: (a.flatten(1) * r).sum(), x)

    # conv, pool, flatten, linear: images in, one row of scores per image out
    model = sequential(conv2d(3, 4, 3, padding=1), lambda h: h.relu(), max_pool2d(2),
                       flatten(), linear(4 * 2 * 2, 3))
    assert max_pool2d(2).parameters() == [] and flatten().parameters() == []
    x = rand_shape(2, 3, 4, 5)
    assert model(x).shape == (2, 3)
    y = rand_shape(2, 3)
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    assert raises(lambda: rand_shape(3, 4, 4).max_pool2d(2), ValueError)
    assert raises(lambda: rand_shape(1, 1, 2, 2).max_pool2d(3), ValueError)
    assert raises(lambda: rand_shape(1, 1, 4, 4).max_pool2d(2, stride=0), ValueError)
    assert raises(lambda: x.flatten(4), IndexError)


def test_avg_pool2d():
    x = tensor([[[[1.0, 3.0, 2.0, 0.0],
                  [4.0, 2.0, 1.0, 5.0],
                  [0.0, 1.0, 9.0, 8.0],
                  [7.0, 6.0, 3.0, 2.0]]]])
    out = x.avg_pool2d(2)
    assert out.shape == (1, 1, 2, 2) and out.data == [2.5, 2.0, 3.5, 5.5]

    # every value in a window gets a quarter of that window's grad
    (out * tensor([[[[1.0, 2.0], [3.0, 4.0]]]])).sum().backward()
    assert x.grad == [0.25, 0.25, 0.5, 0.5,
                      0.25, 0.25, 0.5, 0.5,
                      0.75, 0.75, 1.0, 1.0,
                      0.75, 0.75, 1.0, 1.0]

    # overlapping windows (stride 1): the middle value is in all four windows, a corner in one
    x = tensor([[[[1.0, 2.0, 0.0], [3.0, 9.0, 4.0], [0.0, 5.0, 1.0]]]])
    out = x.avg_pool2d(2, stride=1)
    assert out.data == [3.75, 3.75, 4.25, 4.75]
    out.sum().backward()
    assert x.grad == [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]

    random.seed(23)
    for N, C, H, W, K, stride in [(2, 3, 5, 6, 2, None), (1, 2, 7, 7, 3, 2), (2, 1, 4, 5, 2, 1), (1, 1, 3, 4, 1, None)]:
        x = rand_shape(N, C, H, W)
        S = stride or K
        out = x.avg_pool2d(K, stride)
        OH, OW = (H - K) // S + 1, (W - K) // S + 1
        assert out.shape == (N, C, OH, OW)
        expected = [math.fsum(x.get_element(n, c, i * S + a, j * S + b) for a in range(K) for b in range(K)) / K ** 2
                    for n in range(N) for c in range(C) for i in range(OH) for j in range(OW)]
        assert all(close(a, b) for a, b in zip(out.data, expected))
        r = rand_shape(N, C, OH, OW)
        gradcheck(lambda a: (a.avg_pool2d(K, stride) * r).sum(), x)

    # on one channel it's a conv with a filter of 1 / K^2 everywhere
    x = rand_shape(2, 1, 6, 5)
    mean_filter = tensor([1 / 9] * 9, shape=(1, 1, 3, 3))
    for stride in [1, 2, 3]:
        assert all(close(a, b) for a, b in zip(x.avg_pool2d(3, stride).data, x.conv2d(mean_filter, stride).data))
    # a window as big as the image is the mean of each channel (global average pooling)
    x = rand_shape(2, 3, 4, 4)
    assert all(close(a, b) for a, b in zip(x.avg_pool2d(4).data, x.mean(axis=3).mean(axis=2).data))

    # the layer wraps the op and has no parameters
    pool = avg_pool2d(2)
    assert pool.parameters() == [] and pool(x).data == x.avg_pool2d(2).data
    assert avg_pool2d(3, 1)(x).data == x.avg_pool2d(3, 1).data
    assert repr(pool) == "AvgPool2d(kernel_size=2, stride=2)" and repr(avg_pool2d(3, 1)) == "AvgPool2d(kernel_size=3, stride=1)"
    model = sequential(conv2d(3, 4, 3, padding=1), lambda h: h.relu(), avg_pool2d(2), flatten(), linear(4 * 2 * 2, 3))
    y = rand_shape(2, 3)
    gradcheck(lambda *params: model(x).mse_loss(y), *model.parameters())

    assert raises_with(lambda: rand_shape(3, 4, 4).avg_pool2d(2), ValueError, "(N, C, H, W)")
    assert raises_with(lambda: rand_shape(1, 1, 2, 2).avg_pool2d(3), ValueError, "doesn't fit")
    assert raises_with(lambda: rand_shape(1, 1, 4, 4).avg_pool2d(2, stride=0), ValueError, "stride")
    assert raises_with(lambda: rand_shape(1, 1, 4, 4).avg_pool2d(0), ValueError, "kernel_size")
    assert raises_with(lambda: rand_shape(1, 1, 4, 4).avg_pool2d(True), ValueError, "kernel_size")
    assert raises_with(lambda: rand_shape(1, 1, 4, 4).avg_pool2d(2.0), ValueError, "kernel_size")


def test_rnn_cell():
    random.seed(22)
    cell = rnn_cell(3, 4)
    assert [name for name, _ in cell.named_parameters()] == ["w_x", "w_h", "b"]
    cell.b.data = [random.uniform(-1, 1) for _ in range(4)]

    x, h = rand_tensor(2, 3), rand_tensor(2, 4)
    out = cell(x, h)
    assert out.shape == (2, 4)
    for n in range(2):
        for j in range(4):
            pre = (math.fsum(x.get_element(n, i) * cell.w_x.get_element(i, j) for i in range(3))
                   + math.fsum(h.get_element(n, k) * cell.w_h.get_element(k, j) for k in range(4))
                   + cell.b.data[j])
            assert close(out.get_element(n, j), math.tanh(pre))
    assert cell(x).data == cell(x, tensor([0.0] * 8, shape=(2, 4))).data

    # backprop through time: grads for the shared weights and for every input, 4 steps back
    def run(*sequence):
        h = None
        for x_t in sequence:
            h = cell(x_t, h)
        return h

    sequence = [rand_tensor(2, 3) for _ in range(4)]
    y = rand_tensor(2, 4)
    gradcheck(lambda *params: run(*sequence).mse_loss(y), *cell.parameters())
    gradcheck(lambda *inputs: run(*inputs).mse_loss(y), *sequence)
    assert any(g != 0.0 for g in sequence[0].grad)

    # needs memory: read 6 random +-1 values and say at the end what the first one was.
    # the first value only reaches the output through the hidden state, step after step
    def make(n):
        rows = [[random.choice([-1.0, 1.0]) for _ in range(6)] for _ in range(n)]
        return [tensor([[r[t]] for r in rows]) for t in range(6)], [int(r[0] > 0) for r in rows]

    random.seed(23)
    cell, head = rnn_cell(1, 8), linear(8, 2)
    opt = Adam(cell.parameters() + head.parameters(), lr=0.05)
    for _ in range(100):
        inputs, labels = make(32)
        h = None
        for x_t in inputs:
            h = cell(x_t, h)
        loss = head(h).log_softmax().nll_loss(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    inputs, labels = make(200)
    h = None
    for x_t in inputs:
        h = cell(x_t, h)
    assert head(h).accuracy(labels) == 1.0

    assert raises(lambda: rnn_cell(3, 4)(rand_tensor(2, 2)), ValueError)
    assert raises(lambda: rnn_cell(3, 4)(rand_tensor(2, 3), rand_tensor(3, 4)), ValueError)


def to_rows(t):
    return [t.data[n * t.shape[1]:(n + 1) * t.shape[1]] for n in range(t.shape[0])]


def ref_gate_inputs(cell, x_row, h_row):
    # x_row @ w_x + h_row @ w_h for every column, before any bias
    I, cols = cell.w_x.shape
    H = cell.w_h.shape[0]
    return [math.fsum([x_row[k] * cell.w_x.get_element(k, j) for k in range(I)]
                      + [h_row[k] * cell.w_h.get_element(k, j) for k in range(H)]) for j in range(cols)]


def ref_lstm_step(cell, x, h, c):
    # one step by hand, on lists of rows
    H = cell.w_h.shape[0]
    h_out, c_out = [], []
    for x_row, h_row, c_row in zip(x, h, c):
        pre = [v + b for v, b in zip(ref_gate_inputs(cell, x_row, h_row), cell.b.data)]
        i = [ref_sigmoid(v) for v in pre[:H]]
        f = [ref_sigmoid(v) for v in pre[H:2 * H]]
        g = [math.tanh(v) for v in pre[2 * H:3 * H]]
        o = [ref_sigmoid(v) for v in pre[3 * H:]]
        c_new = [f[j] * c_row[j] + i[j] * g[j] for j in range(H)]
        c_out.append(c_new)
        h_out.append([o[j] * math.tanh(c_new[j]) for j in range(H)])
    return h_out, c_out


def test_lstm_cell():
    random.seed(42)
    cell = lstm_cell(3, 4)
    assert [name for name, _ in cell.named_parameters()] == ["w_x", "w_h", "b"]
    assert cell.w_x.shape == (3, 16) and cell.w_h.shape == (4, 16) and cell.b.shape == (1, 16)
    # gates in the order i, f, g, o, and only the forget gate's bias starts at 1
    assert cell.b.data == [0.0] * 4 + [1.0] * 4 + [0.0] * 8
    assert repr(cell) == "LSTMCell(3, 4)"
    cell.b.data = [random.uniform(-1, 1) for _ in range(16)]

    x, h, c = rand_tensor(2, 3), rand_tensor(2, 4), rand_tensor(2, 4)
    h_new, c_new = cell(x, (h, c))
    assert h_new.shape == c_new.shape == (2, 4)
    ref_h, ref_c = ref_lstm_step(cell, to_rows(x), to_rows(h), to_rows(c))
    for n in range(2):
        for j in range(4):
            assert close(h_new.get_element(n, j), ref_h[n][j])
            assert close(c_new.get_element(n, j), ref_c[n][j])
    zeros = tensor([0.0] * 8, shape=(2, 4))
    h0, c0 = cell(x)
    h1, c1 = cell(x, (zeros, zeros))
    assert h0.data == h1.data and c0.data == c1.data

    def run(cell, sequence, state=None):
        for x_t in sequence:
            state = cell(x_t, state)
        return state

    # backprop through time, 4 steps back: grads for the weights, every input and the
    # starting state. the loss reads c as well as h, so both paths out of a step are checked
    sequence = [rand_tensor(2, 3) for _ in range(4)]
    y = rand_tensor(2, 4)
    loss = lambda state: state[0].mse_loss(y) + (state[1] * state[1]).mean()
    gradcheck(lambda *params: loss(run(cell, sequence)), *cell.parameters())
    gradcheck(lambda *inputs: loss(run(cell, inputs)), *sequence)
    gradcheck(lambda h, c: loss(run(cell, sequence, (h, c))), h, c)
    assert any(g != 0.0 for g in sequence[0].grad)

    # with the forget gate wide open and the input gate shut, c goes through every step
    # unchanged, and so does its gradient: dc_t/dc_(t-1) = f = 1
    cell.b.data = [-50.0] * 4 + [50.0] * 4 + [0.0] * 8
    c_start = rand_tensor(2, 4)
    _, c_end = run(cell, sequence, (rand_tensor(2, 4), c_start))
    assert all(abs(a - b) < 1e-12 for a, b in zip(c_end.data, c_start.data))
    c_end.sum().backward()
    assert all(abs(g - 1.0) < 1e-12 for g in c_start.grad)

    # the memory task from the rnn_cell test: read 6 random +-1 values, then say what the
    # first one was
    def make(n):
        rows = [[random.choice([-1.0, 1.0]) for _ in range(6)] for _ in range(n)]
        return [tensor([[r[t]] for r in rows]) for t in range(6)], [int(r[0] > 0) for r in rows]

    random.seed(43)
    cell, head = lstm_cell(1, 4), linear(4, 2)
    opt = Adam(cell.parameters() + head.parameters(), lr=0.05)
    for _ in range(60):
        inputs, labels = make(32)
        h, _ = run(cell, inputs)
        loss = head(h).log_softmax().nll_loss(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    inputs, labels = make(200)
    h, _ = run(cell, inputs)
    assert head(h).accuracy(labels) == 1.0

    cell, x = lstm_cell(3, 4), rand_tensor(2, 3)
    assert raises_with(lambda: cell(rand_tensor(2, 2)), ValueError, "input shape (N, 3)")
    assert raises_with(lambda: cell(x, (rand_tensor(3, 4), rand_tensor(2, 4))), ValueError, "hidden shape (2, 4)")
    assert raises_with(lambda: cell(x, (rand_tensor(2, 4), rand_tensor(2, 5))), ValueError, "cell shape (2, 4)")


def ref_row_matmul(row, w):
    return [math.fsum(row[k] * w.get_element(k, j) for k in range(len(row))) for j in range(w.shape[1])]


def ref_gru_step(cell, x, h):
    # one step by hand, on lists of rows
    H = cell.w_h.shape[0]
    out = []
    for x_row, h_row in zip(x, h):
        gx = [v + b for v, b in zip(ref_row_matmul(x_row, cell.w_x), cell.b_x.data)]
        gh = [v + b for v, b in zip(ref_row_matmul(h_row, cell.w_h), cell.b_h.data)]
        r = [ref_sigmoid(gx[j] + gh[j]) for j in range(H)]
        z = [ref_sigmoid(gx[H + j] + gh[H + j]) for j in range(H)]
        n = [math.tanh(gx[2 * H + j] + r[j] * gh[2 * H + j]) for j in range(H)]
        out.append([(1 - z[j]) * n[j] + z[j] * h_row[j] for j in range(H)])
    return out


def test_gru_cell():
    random.seed(43)
    cell = gru_cell(3, 4)
    assert [name for name, _ in cell.named_parameters()] == ["w_x", "w_h", "b_x", "b_h"]
    assert cell.w_x.shape == (3, 12) and cell.w_h.shape == (4, 12)
    assert cell.b_x.data == cell.b_h.data == [0.0] * 12 and cell.b_x.shape == cell.b_h.shape == (1, 12)
    assert repr(cell) == "GRUCell(3, 4)"
    # different biases on the two sides, so a mix-up between them shows up in n
    cell.b_x.data = [random.uniform(-1, 1) for _ in range(12)]
    cell.b_h.data = [random.uniform(-1, 1) for _ in range(12)]

    x, h = rand_tensor(2, 3), rand_tensor(2, 4)
    out = cell(x, h)
    assert out.shape == (2, 4)
    ref = ref_gru_step(cell, to_rows(x), to_rows(h))
    for n in range(2):
        for j in range(4):
            assert close(out.get_element(n, j), ref[n][j])
    assert cell(x).data == cell(x, tensor([0.0] * 8, shape=(2, 4))).data

    def run(cell, sequence, h=None):
        for x_t in sequence:
            h = cell(x_t, h)
        return h

    # backprop through time, 4 steps back: grads for the weights, every input and the starting h
    sequence = [rand_tensor(2, 3) for _ in range(4)]
    y = rand_tensor(2, 4)
    gradcheck(lambda *params: run(cell, sequence).mse_loss(y), *cell.parameters())
    gradcheck(lambda *inputs: run(cell, inputs).mse_loss(y), *sequence)
    gradcheck(lambda h: run(cell, sequence, h).mse_loss(y), h)
    assert any(g != 0.0 for g in sequence[0].grad)

    # with the update gate shut (z = 1), h goes through every step unchanged, and so does
    # its gradient: dh_t/dh_(t-1) = z = 1
    cell.b_x.data = [0.0] * 4 + [50.0] * 4 + [0.0] * 4
    h_start = rand_tensor(2, 4)
    h_end = run(cell, sequence, h_start)
    assert all(abs(a - b) < 1e-12 for a, b in zip(h_end.data, h_start.data))
    h_end.sum().backward()
    assert all(abs(g - 1.0) < 1e-12 for g in h_start.grad)

    # the memory task from the rnn_cell test: read 6 random +-1 values, then say what the
    # first one was
    def make(n):
        rows = [[random.choice([-1.0, 1.0]) for _ in range(6)] for _ in range(n)]
        return [tensor([[r[t]] for r in rows]) for t in range(6)], [int(r[0] > 0) for r in rows]

    random.seed(44)
    cell, head = gru_cell(1, 4), linear(4, 2)
    opt = Adam(cell.parameters() + head.parameters(), lr=0.05)
    for _ in range(60):
        inputs, labels = make(32)
        loss = head(run(cell, inputs)).log_softmax().nll_loss(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    inputs, labels = make(200)
    assert head(run(cell, inputs)).accuracy(labels) == 1.0

    cell, x = gru_cell(3, 4), rand_tensor(2, 3)
    assert raises_with(lambda: cell(rand_tensor(2, 2)), ValueError, "input shape (N, 3)")
    assert raises_with(lambda: cell(x, rand_tensor(3, 4)), ValueError, "hidden shape (2, 4)")


def ref_attention(q, k, v, causal):
    # one query row at a time, with math.exp
    T_q, d = q.shape
    T_k, d_v = v.shape
    out = []
    for i in range(T_q):
        visible = range(i + 1) if causal else range(T_k)
        scores = [math.fsum(q.get_element(i, c) * k.get_element(j, c) for c in range(d)) / math.sqrt(d)
                  for j in visible]
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = math.fsum(weights)
        out += [math.fsum(w * v.get_element(j, c) for w, j in zip(weights, visible)) / total for c in range(d_v)]
    return out


def test_scaled_dot_product_attention():
    random.seed(24)
    q, k, v = rand_shape(3, 4, low=-2, high=2), rand_shape(5, 4, low=-2, high=2), rand_shape(5, 2)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (3, 2)
    assert all(close(a, b) for a, b in zip(out.data, ref_attention(q, k, v, causal=False)))

    # keys that all look the same get equal weight, so the output is the mean value row
    same_keys = tensor([[1.0, 0.5, -1.0, 2.0]] * 5)
    for row in range(3):
        for c in range(2):
            mean = math.fsum(v.get_element(j, c) for j in range(5)) / 5
            assert close(scaled_dot_product_attention(q, same_keys, v).get_element(row, c), mean)
    # a query that matches one key far better than the rest copies that key's value
    keys = tensor([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
    values = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    picked = scaled_dot_product_attention(tensor([[0.0, 10.0]]), keys, values)
    assert abs(picked.data[0] - 3.0) < 1e-12 and abs(picked.data[1] - 4.0) < 1e-12

    # causal: position i averages only values 0..i, so position 0 returns value 0 exactly
    T = 4
    q, k, v = rand_shape(T, 3), rand_shape(T, 3), rand_shape(T, 2)
    out = scaled_dot_product_attention(q, k, v, causal=True)
    assert all(close(a, b) for a, b in zip(out.data, ref_attention(q, k, v, causal=True)))
    assert close(out.data[0], v.data[0]) and close(out.data[1], v.data[1])

    # changing the last key and value can't change earlier outputs, and earlier outputs
    # send them no gradient
    k2 = tensor(k.data[:9] + [5.0, -5.0, 5.0], shape=(T, 3))
    v2 = tensor(v.data[:6] + [100.0, -100.0], shape=(T, 2))
    changed = scaled_dot_product_attention(q, k2, v2, causal=True)
    assert changed.data[:6] == out.data[:6] and changed.data[6:] != out.data[6:]
    early = scaled_dot_product_attention(q, k, v, causal=True)[:3]
    early.sum().backward()
    assert k.grad[9:] == [0.0] * 3 and v.grad[6:] == [0.0] * 2
    assert any(g != 0.0 for g in k.grad[:9])

    # leading dims: each (batch, head) slice is attended separately
    q, k, v = rand_shape(2, 2, 4, 3), rand_shape(2, 2, 4, 3), rand_shape(2, 2, 4, 2)
    for causal in [False, True]:
        out = scaled_dot_product_attention(q, k, v, causal=causal)
        assert out.shape == (2, 2, 4, 2)
        for b in range(2):
            for h in range(2):
                one = scaled_dot_product_attention(q[b, h], k[b, h], v[b, h], causal=causal)
                assert out[b, h].data == one.data
        r = rand_shape(2, 2, 4, 2)
        gradcheck(lambda a, b, c: (scaled_dot_product_attention(a, b, c, causal=causal) * r).sum(), q, k, v)

    q, k, v = rand_shape(3, 4), rand_shape(5, 4), rand_shape(5, 2)
    assert raises(lambda: scaled_dot_product_attention(q, rand_shape(5, 3), v), ValueError)
    assert raises(lambda: scaled_dot_product_attention(q, k, rand_shape(4, 2)), ValueError)
    assert raises(lambda: scaled_dot_product_attention(q, k, v, causal=True), ValueError)
    assert raises(lambda: scaled_dot_product_attention(rand_shape(2, 3, 4), rand_shape(1, 3, 4), rand_shape(1, 3, 4)), ValueError)


def test_linear_with_leading_dims():
    random.seed(25)
    layer = linear(4, 3)
    layer.b.data = [0.5, -1.0, 2.0]
    x = rand_shape(2, 5, 4)
    out = layer(x)
    assert out.shape == (2, 5, 3)
    # same as applying the layer to each (5, 4) slice, or to every row on its own
    for n in range(2):
        assert out[n].data == layer(x[n]).data
    assert layer(x[1, 2]).shape == (3,) and layer(x[1, 2]).data == out[1, 2].data
    # 2-d inputs keep the plain matmul + bias graph
    assert layer(x[0])._op == "+"

    y = rand_shape(2, 5, 3)
    gradcheck(lambda a, w, b: layer(a).mse_loss(y), x, layer.w, layer.b)
    assert raises(lambda: layer(rand_shape(2, 5, 3)), ValueError)


def ref_project(layer, rows):
    return [[math.fsum(r[i] * layer.w.get_element(i, j) for i in range(len(r))) + layer.b.data[j]
             for j in range(layer.w.shape[1])] for r in rows]


def ref_mha(mha, rows, causal):
    # one sequence, a list of rows: project by hand, attend in each head's slice of the
    # features, put the heads side by side and project again
    H = mha.num_heads
    head_dim = len(rows[0]) // H
    q, k, v = ref_project(mha.q_proj, rows), ref_project(mha.k_proj, rows), ref_project(mha.v_proj, rows)
    heads = []
    for h in range(H):
        cut = lambda m: tensor([r[h * head_dim:(h + 1) * head_dim] for r in m])
        heads.append(ref_attention(cut(q), cut(k), cut(v), causal))
    merged = [[heads[h][t * head_dim + c] for h in range(H) for c in range(head_dim)] for t in range(len(rows))]
    return ref_project(mha.out_proj, merged)


def test_multi_head_attention():
    random.seed(26)
    B, T, D, H = 2, 4, 6, 3
    mha = multi_head_attention(D, num_heads=H)
    assert [name for name, _ in mha.named_parameters()] == [
        f"{proj}.{p}" for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] for p in ["w", "b"]]
    for p in mha.parameters():
        p.data = [random.uniform(-1, 1) for _ in p.data]
    x = rand_shape(B, T, D)

    outputs = {}
    for causal in [False, True]:
        out = mha(x, causal=causal)
        assert out.shape == (B, T, D)
        for b in range(B):
            rows = [[x.get_element(b, t, c) for c in range(D)] for t in range(T)]
            expected = ref_mha(mha, rows, causal)
            for t in range(T):
                for c in range(D):
                    assert close(out.get_element(b, t, c), expected[t][c], abs_tol=1e-12)
        outputs[causal] = out

    # causal: changing the last position can't change the outputs before it
    later = tensor(x.data[:], shape=x.shape)
    for b in range(B):
        for c in range(D):
            later.data[(b * T + T - 1) * D + c] += 1.0
    changed = mha(later, causal=True)
    for b in range(B):
        assert changed[b, :T - 1].data == outputs[True][b, :T - 1].data
        assert changed[b, T - 1].data != outputs[True][b, T - 1].data
    assert mha(later)[0, 0].data != outputs[False][0, 0].data

    small = multi_head_attention(4, num_heads=2)
    x, y = rand_shape(2, 3, 4), rand_shape(2, 3, 4)
    for causal in [False, True]:
        gradcheck(lambda a, *params: small(a, causal).mse_loss(y), x, *small.parameters())

    assert raises(lambda: multi_head_attention(6, num_heads=4), ValueError)
    assert raises(lambda: small(rand_shape(3, 4)), ValueError)
    assert raises(lambda: small(rand_shape(2, 3, 5)), ValueError)


def ref_layer_norm(ln, rows):
    out = []
    for r in rows:
        mean = math.fsum(r) / len(r)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in r) / len(r) + ln.eps)
        out.append([(v - mean) / std * ln.gamma.data[j] + ln.beta.data[j] for j, v in enumerate(r)])
    return out


def test_transformer_block():
    random.seed(32)
    B, T, D, H = 2, 4, 6, 2
    block = transformer_block(D, num_heads=H, hidden=8)
    assert [name for name, _ in block.named_parameters()] == (
        ["ln1.gamma", "ln1.beta"]
        + [f"attn.{proj}.{p}" for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] for p in ["w", "b"]]
        + ["ln2.gamma", "ln2.beta"]
        + [f"mlp.layers.{i}.{p}" for i in range(2) for p in ["w", "b"]])
    assert block.mlp.layers[0].w.shape == (6, 8)
    assert transformer_block(8, num_heads=2).mlp.layers[0].w.shape == (8, 32)
    for p in block.parameters():
        p.data = [random.uniform(-1, 1) for _ in p.data]
    x = rand_shape(B, T, D)

    k = math.sqrt(2 / math.pi)
    gelu = lambda v: 0.5 * v * (1 + math.tanh(k * (v + 0.044715 * v ** 3)))
    add = lambda m, n: [[u + w for u, w in zip(r, s)] for r, s in zip(m, n)]

    # reference for one sequence, both residual steps written out by hand
    def ref_block(rows, causal):
        rows = add(rows, ref_mha(block.attn, ref_layer_norm(block.ln1, rows), causal))
        hidden = ref_project(block.mlp.layers[0], ref_layer_norm(block.ln2, rows))
        return add(rows, ref_project(block.mlp.layers[1], [[gelu(v) for v in r] for r in hidden]))

    outputs = {}
    for causal in [False, True]:
        out = block(x, causal=causal)
        assert out.shape == (B, T, D)
        for b in range(B):
            rows = [[x.get_element(b, t, c) for c in range(D)] for t in range(T)]
            expected = ref_block(rows, causal)
            for t in range(T):
                for c in range(D):
                    assert close(out.get_element(b, t, c), expected[t][c], abs_tol=1e-12)
        outputs[causal] = out

    # causal: the mlp works on each position alone, so changing the last position still
    # can't reach the outputs before it. the change differs per feature, since layer norm
    # would erase the same shift added to every feature
    later = tensor(x.data[:], shape=x.shape)
    for b in range(B):
        for c in range(D):
            later.data[(b * T + T - 1) * D + c] += random.uniform(-1, 1)
    changed = block(later, causal=True)
    for b in range(B):
        assert changed[b, :T - 1].data == outputs[True][b, :T - 1].data
        assert changed[b, T - 1].data != outputs[True][b, T - 1].data
    assert block(later)[0, 0].data != outputs[False][0, 0].data

    # with the last weights of both sub-layers at zero they add nothing: the block passes x
    # through unchanged and the gradient reaches x along the residual path alone. the zeroed
    # weights still get gradients, so training can move them away from zero
    for layer in [block.attn.out_proj, block.mlp.layers[1]]:
        layer.w.data = [0.0] * len(layer.w.data)
        layer.b.data = [0.0] * len(layer.b.data)
    x = rand_shape(B, T, D)
    out = block(x, causal=True)
    assert out.data == x.data
    out.sum().backward()
    assert x.grad == [1.0] * len(x.data)
    assert any(g != 0 for g in block.attn.out_proj.w.grad)
    assert any(g != 0 for g in block.mlp.layers[1].w.grad)

    small = transformer_block(4, num_heads=2, hidden=6)
    for p in small.parameters():
        p.data = [random.uniform(-1, 1) for _ in p.data]
    x, y = rand_shape(2, 3, 4), rand_shape(2, 3, 4)
    for causal in [False, True]:
        gradcheck(lambda a, *params: small(a, causal).mse_loss(y), x, *small.parameters())

    assert raises(lambda: transformer_block(4, num_heads=3), ValueError)
    for bad in [0, -1, 2.5, True]:
        assert raises(lambda: transformer_block(4, num_heads=2, hidden=bad), ValueError)
    assert raises(lambda: small(rand_shape(3, 4)), ValueError)
    assert raises(lambda: small(rand_shape(2, 3, 5)), ValueError)


def ref_sample(row, temperature, top_k, rng):
    # math.exp weights that aren't normalized, and u scaled up to their total instead
    V = len(row)
    keep = sorted(range(V), key=lambda i: (-row[i], i))[:top_k or V]
    top = max(row[i] for i in keep)
    weights = [math.exp((row[i] - top) / temperature) if i in keep else 0.0 for i in range(V)]
    u = rng.random() * math.fsum(weights)
    end = 0.0
    for i, w in enumerate(weights):
        end += w
        if u < end:
            return i


def test_sample():
    random.seed(33)
    z = [1.0, 2.0, 0.5, -1.0, 0.0, 2.0]
    probs = [math.exp(v) / math.fsum(math.exp(w) for w in z) for v in z]

    # each id comes up as often as its softmax probability says. the count of id i out of
    # N draws is binomial, so it stays within 5 standard deviations of N * p_i
    N = 10000
    picks = sample(tensor([z] * N))
    assert len(picks) == N and all(isinstance(i, int) for i in picks)
    for i, p in enumerate(probs):
        assert abs(picks.count(i) - N * p) < 5 * math.sqrt(N * p * (1 - p)), (i, picks.count(i), N * p)

    # draw for draw the same as the reference, given the same random numbers
    for temperature in [0.3, 1.0, 2.5]:
        for top_k in [None, 1, 2, 3, 6, 10]:
            got = sample(tensor([z] * 200), temperature, top_k, rng=random.Random(5))
            ref_rng = random.Random(5)
            assert got == [ref_sample(z, temperature, top_k, ref_rng) for _ in range(200)], (temperature, top_k)

    # top_k: ids 1 and 5 tie for the top score, and on ties the lower index is kept
    assert set(sample(tensor([z] * 200), top_k=2)) == {1, 5}
    assert sample(tensor([z] * 50), top_k=1) == [1] * 50

    # temperature 0 takes the top score, the first on ties, without using up a random number
    state = random.getstate()
    assert sample(tensor(z), temperature=0) == 1
    assert sample(tensor([z, [0.0, -1.0, 3.0, 1.0, 0.0, 0.0]]), temperature=0) == [1, 2]
    assert random.getstate() == state

    # every row draws from its own scores, and a -inf score is never picked
    inf = float("-inf")
    picks = sample(tensor([[inf, inf, 0.0], [0.0, inf, inf], [0.0, inf, 0.0]] * 100))
    assert picks[0::3] == [2] * 100 and picks[1::3] == [0] * 100 and set(picks[2::3]) == {0, 2}

    # these probabilities add up to 0.9999999999999998 after rounding, so the largest u that
    # random() gives lands past the end. it goes to the last id that can be picked
    class largest_u:
        def random(self):
            return 1 - 2 ** -53
    assert sample(tensor([2.9, -2.8, -2.9, inf]), rng=largest_u()) == 2

    for bad in [-0.5, True, "1", float("inf"), float("nan")]:
        assert raises(lambda: sample(tensor(z), temperature=bad), ValueError)
    for bad in [0, -2, 1.5, True]:
        assert raises(lambda: sample(tensor(z), top_k=bad), ValueError)
    assert raises(lambda: sample(tensor(2.0)), ValueError)
    assert raises(lambda: sample(tensor([[z]])), ValueError)
    assert raises(lambda: sample(tensor([[0.0, 1.0], [inf, inf]])), ValueError)


def test_no_grad():
    random.seed(0)
    x = tensor([[0.5, -1.0, 2.0], [1.5, 0.3, -0.7]])
    w = tensor([[0.2, -0.4], [1.1, 0.6], [-0.3, 0.9]])
    f = lambda: ((x @ w).tanh() * 2.0 + 1.0).log_softmax().sum()
    expected = f().data

    # same values, but no graph: the result has no inputs and backward() reaches nothing
    with no_grad():
        assert not is_grad_enabled()
        out = f()
    assert is_grad_enabled()
    assert out.data == expected
    assert out._prev == set() and out._op == ''
    out._backward()
    out.backward()
    assert x.grad == [0.0] * 6 and w.grad == [0.0] * 6

    # a result made inside is a constant to a graph built later
    with no_grad():
        c = x * 3.0
    y = tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 4.0]])
    (c * y).sum().backward()
    assert y.grad == c.data
    assert x.grad == [0.0] * 6

    # a graph built outside still backpropagates inside
    y.grad = [0.0] * 6
    loss = (x * y).sum()
    with no_grad():
        loss.backward()
    assert x.grad == y.data and y.grad == x.data

    # the old setting comes back after nested blocks and after an error
    with no_grad():
        with no_grad():
            pass
        assert not is_grad_enabled()
    assert is_grad_enabled()
    try:
        with no_grad():
            raise KeyError("boom")
    except KeyError:
        pass
    assert is_grad_enabled()

    # switching must not write to the tensor class, see the comment on _grad_enabled
    before = dict(vars(tensor))
    with no_grad():
        assert dict(vars(tensor)) == before

    # without the graph, a result doesn't keep its inputs alive
    a = tensor([1.0, 2.0])
    kept = weakref.ref(a)
    b = (a * 2.0).exp()
    del a
    gc.collect()
    assert kept() is not None
    a = tensor([1.0, 2.0])
    gone = weakref.ref(a)
    with no_grad():
        b = (a * 2.0).exp()
    del a
    gc.collect()
    assert gone() is None
    assert all(close(v, math.exp(2.0 * k)) for v, k in zip(b.data, [1.0, 2.0]))

    # a whole model gives the same output, and its parameters get no grads
    block = transformer_block(8, 2)
    h = tensor([[[random.uniform(-1, 1) for _ in range(8)] for _ in range(3)] for _ in range(2)])
    expected = block(h, causal=True).data
    with no_grad():
        out = block(h, causal=True)
    assert out.data == expected and out._prev == set()
    out.sum().backward()
    assert all(g == 0.0 for p in block.parameters() for g in p.grad)


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
