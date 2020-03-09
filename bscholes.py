import sys
import math
import numpy as np
from numba import jit

@jit
def cnd_numba(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val

@jit
def black_scholes_numba(stockPrice, optionStrike,
                        optionYears, Riskfree, Volatility):
    callResult = np.empty_like(stockPrice)
    putResult = np.empty_like(stockPrice)
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    for i in range(len(S)):
        sqrtT = math.sqrt(T[i])
        d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        cndd1 = cnd_numba(d1)
        cndd2 = cnd_numba(d2)

        expRT = math.exp((-1. * R) * T[i])
        callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))

    return callResult, putResult

RISKFREE = 0.02
VOLATILITY = 0.30


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


def input_generator():
    for dtype in [np.float64, np.float32]:
        for OPT_N in [1000, 100000, 1000000, 4000000]:
            category = (np.dtype(dtype).name, )

            stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0).astype(dtype)
            optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0).astype(dtype)
            optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0).astype(dtype)

            yield dict(category=category, x=OPT_N, input_args=(stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY), input_kwargs={})


def validator(input_args, input_kwargs, output):
    actual_call, actual_put = output
    
    expected_call, expected_put = black_scholes_numba(*input_args, **input_kwargs)

    np.testing.assert_allclose(actual_call, expected_call, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_put, expected_put, rtol=1e-5, atol=1e-5)