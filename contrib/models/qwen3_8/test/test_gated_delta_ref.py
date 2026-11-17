# SPDX-License-Identifier: Apache-2.0
"""Pure-Python check of one Gated DeltaNet step.

Mirrors ``recurrent_gated_delta_rule`` for a single head and a single token,
so it can run without torch. The torch module must stay in lockstep with this
arithmetic: L2-normalize q and k, scale q by 1/sqrt(Dk), decay, then the
delta-rule rank-1 update.
"""


def l2(vec, eps=1e-6):
    norm = sum(x * x for x in vec) ** 0.5
    inv = 1.0 / (norm * norm + eps) ** 0.5
    return [x * inv for x in vec]


def step(q, k, v, g, beta, state):
    """state is row-major list of lists [Dk][Dv]. Returns (out, new_state)."""
    q = l2(q)
    k = l2(k)
    scale = len(q) ** -0.5
    q = [x * scale for x in q]
    decay = 2.718281828459045 ** g
    state = [[s * decay for s in row] for row in state]
    # kv_mem_d = sum_i state[i][d] * k[i]
    kv = [sum(state[i][d] * k[i] for i in range(len(k))) for d in range(len(v))]
    delta = [(v[d] - kv[d]) * beta for d in range(len(v))]
    state = [
        [state[i][d] + k[i] * delta[d] for d in range(len(v))] for i in range(len(k))
    ]
    out = [sum(state[i][d] * q[i] for i in range(len(q))) for d in range(len(v))]
    return out, state


def test_zero_state_one_step():
    # k = [0, 1] is already unit length. v = [1, 0], beta = 1, g = 0 (decay 1).
    state0 = [[0.0, 0.0], [0.0, 0.0]]
    out, state = step([3.0, 4.0], [0.0, 1.0], [1.0, 0.0], 0.0, 1.0, state0)
    # k is L2-normalized with eps, so the write is k_hat^T * v, not exactly [0, 1].
    k_hat = l2([0.0, 1.0])
    assert abs(state[0][0]) < 1e-9 and abs(state[0][1]) < 1e-9
    assert abs(state[1][0] - k_hat[1]) < 1e-9
    assert abs(state[1][1]) < 1e-9
    q = [x * (2 ** -0.5) for x in l2([3.0, 4.0])]
    assert abs(out[0] - q[1] * state[1][0]) < 1e-9
    assert abs(out[1]) < 1e-9


def test_decay_shrinks_state_before_write():
    state0 = [[2.0, 0.0], [0.0, 0.0]]
    # beta = 0 writes nothing. g = log(0.5) so the state is halved.
    import math

    _out, state = step([1.0, 0.0], [1.0, 0.0], [0.0, 0.0], math.log(0.5), 0.0, state0)
    assert abs(state[0][0] - 1.0) < 1e-9
    assert state[0][1] == 0.0
    assert state[1] == [0.0, 0.0]


if __name__ == "__main__":
    test_zero_state_one_step()
    test_decay_shrinks_state_before_write()
    print("ok")
