#!/usr/bin/env python3
# A discrete Hidden Markov Model: the Viterbi algorithm (most-likely state path)
# and forward-backward (posterior state probabilities), from scratch with numpy.
import numpy as np


def viterbi(obs, init, trans, emit):
    # Most-likely hidden-state sequence. Works in log space to avoid underflow.
    n, S = len(obs), len(init)
    logI, logT, logE = np.log(init + 1e-300), np.log(trans + 1e-300), np.log(emit + 1e-300)
    delta = np.full((n, S), -np.inf); psi = np.zeros((n, S), int)
    delta[0] = logI + logE[:, obs[0]]
    for t in range(1, n):
        scores = delta[t-1][:, None] + logT            # (S_prev, S_cur)
        psi[t] = scores.argmax(0)
        delta[t] = scores.max(0) + logE[:, obs[t]]
    path = np.zeros(n, int); path[-1] = delta[-1].argmax()
    for t in range(n-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path


def forward_backward(obs, init, trans, emit):
    # Posterior P(state | whole sequence) at each position, with scaling.
    n, S = len(obs), len(init)
    alpha = np.zeros((n, S)); c = np.zeros(n)
    alpha[0] = init * emit[:, obs[0]]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, n):
        alpha[t] = (alpha[t-1] @ trans) * emit[:, obs[t]]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.zeros((n, S)); beta[-1] = 1.0
    for t in range(n-2, -1, -1):
        beta[t] = (trans @ (emit[:, obs[t+1]] * beta[t+1])) / c[t+1]
    gamma = alpha * beta
    return gamma / gamma.sum(1, keepdims=True)


def simulate(init, trans, emit, n, seed=0):
    rng = np.random.default_rng(seed)
    S = len(init); states = np.zeros(n, int); obs = np.zeros(n, int)
    states[0] = rng.choice(S, p=init)
    for t in range(1, n):
        states[t] = rng.choice(S, p=trans[states[t-1]])
    for t in range(n):
        obs[t] = rng.choice(emit.shape[1], p=emit[states[t]])
    return states, obs


if __name__ == "__main__":
    init = np.array([0.9, 0.1])
    trans = np.array([[0.999, 0.001], [0.02, 0.98]])
    emit = np.array([[0.3, 0.2, 0.2, 0.3], [0.15, 0.35, 0.35, 0.15]])
    st, ob = simulate(init, trans, emit, 3000, seed=1)
    path = viterbi(ob, init, trans, emit)
    print("Viterbi accuracy:", round((path == st).mean(), 3))
