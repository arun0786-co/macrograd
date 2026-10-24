import math
import random

from engine import tensor


def close(a, b, rel=1e-9, abs_tol=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def test_exp_matches_math():
    for x in [-700, -50, -10, -5, -1, -1e-3, 0, 1e-3, 1, 5, 10, 50, 700]:
        assert close(tensor._taylor_exp(x), math.exp(x)), x
    assert tensor._taylor_exp(1000) == float('inf')
    assert tensor._taylor_exp(-1000) == 0.0


def test_log_matches_math():
    for x in [1e-300, 1e-8, 1e-4, 0.01, 0.5, 1.0, 2.0, 100.0, 1e4, 1e300]:
        assert close(tensor._taylor_log(x), math.log(x)), x


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
