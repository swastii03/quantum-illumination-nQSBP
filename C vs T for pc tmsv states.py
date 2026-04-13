import numpy as np
import matplotlib.pyplot as plt

def factorialValue(n):
    return np.math.factorial(n)

def tmsvState(lambdaVal, dim):
    state = np.zeros(dim * dim, dtype=complex)
    norm = np.sqrt(1 - lambdaVal**2)
    for n in range(dim):
        state[n * dim + n] = norm * (lambdaVal**n)
    return state

def singleSideCatalysis(lambdaVal, tVal, dim):
    state = np.zeros(dim * dim, dtype=complex)
    norm = 0.0
    for n in range(dim):
        gCoef = tVal**((n - 1) / 2) * (tVal * (n + 1) - n)
        coeff = np.sqrt(1 - lambdaVal**2) * (lambdaVal**n) * gCoef
        state[n * dim + n] = coeff
        norm += np.abs(coeff)**2
    state /= np.sqrt(norm)
    return state, norm

def bisideCatalysis(lambdaVal, tVal, dim):
    state = np.zeros(dim * dim, dtype=complex)
    norm = 0.0
    for n in range(dim):
        gCoef = tVal**((n - 1) / 2) * (tVal * (n + 1) - n)
        coeff = np.sqrt(1 - lambdaVal**2) * (lambdaVal**n) * gCoef**2
        state[n * dim + n] = coeff
        norm += np.abs(coeff)**2
    state /= np.sqrt(norm)
    return state, norm

def reducedStateModeA(state, dim):
    probs = np.zeros(dim)
    for n in range(dim):
        probs[n] = np.abs(state[n * dim + n])**2
    return probs

def coherenceMeasure(probs):
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps))

dimVal = 12
lambdaVal = 0.5
tVals = np.linspace(0.0, 0.5, 25)

origState = tmsvState(lambdaVal, dimVal)
origProbs = reducedStateModeA(origState, dimVal)
origCoherence = coherenceMeasure(origProbs)

cohSingle, pSingle = [], []
cohBiside, pBiside = [], []

for tVal in tVals:
    stateSingle, normSingle = singleSideCatalysis(lambdaVal, tVal, dimVal)
    probsSingle = reducedStateModeA(stateSingle, dimVal)
    cohSingle.append(coherenceMeasure(probsSingle))
    pSingle.append(normSingle)

    stateBiside, normBiside = bisideCatalysis(lambdaVal, tVal, dimVal)
    probsBiside = reducedStateModeA(stateBiside, dimVal)
    cohBiside.append(coherenceMeasure(probsBiside))
    pBiside.append(normBiside)

plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.plot(tVals, [origCoherence] * len(tVals), 'o-k', label="Original TMSV")
plt.plot(tVals, cohSingle, '^-r', label="Single-Side PC")
plt.plot(tVals, cohBiside, 's--b', label="Biside PC")
plt.xlabel("Transmittance T")
plt.ylabel("Coherence")
plt.title("Coherence vs T (λ = 0.5)")
plt.xlim(0, 0.4)
plt.ylim(0, 1.5)
plt.xticks(np.arange(0, 0.41, 0.1))
plt.yticks(np.arange(0, 1.51, 0.5))
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tVals, pSingle, '^-r', label="Single-Side PC Success")
plt.plot(tVals, pBiside, 's--b', label="Biside PC Success")
plt.xlabel("Transmittance T")
plt.ylabel("Success Probability")
plt.title("Success Probability vs T (λ = 0.5)")
plt.xlim(0, 0.4)
plt.xticks(np.arange(0, 0.41, 0.1))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
