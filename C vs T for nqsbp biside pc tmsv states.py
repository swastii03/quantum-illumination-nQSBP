import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    return np.math.factorial(n)

def safeT(T):
    return T if T > 1e-8 else 1e-8

def reducedStateModeA(state, dim):
    probs = np.zeros(dim)
    for n in range(dim):
        index = n * dim + n
        probs[n] = np.abs(state[index])**2
    return probs

def coherenceFromProbabilities(probs):
    epsilon = 1e-12
    return -np.sum(probs * np.log(probs + epsilon))

def tmsvState(lmbda, dim):
    state = np.zeros(dim * dim, dtype=complex)
    norm = np.sqrt(1 - lmbda**2)
    for n in range(dim):
        coeff = norm * (lmbda ** n)
        state[n * dim + n] = coeff
    return state

def applySingleSideCatalysisTmsv(lmbda, T, dim):
    state = np.zeros(dim * dim, dtype=complex)
    normalization = 0.0
    for n in range(dim):
        if T == 0 and n > 0:
            gTn = 0
        else:
            gTn = T**((n-1)/2) * (T*(n + 1) - n)
        coeff = np.sqrt(1 - lmbda**2) * (lmbda**n) * gTn
        state[n * dim + n] = coeff
        normalization += np.abs(coeff)**2
    state /= np.sqrt(normalization)
    return state, normalization

def applyBisideCatalysisTmsv(lmbda, T, dim):
    state = np.zeros(dim * dim, dtype=complex)
    normalization = 0.0
    for n in range(dim):
        if T == 0 and n > 0:
            gTn = 0
        else:
            gTn = T**((n-1)/2) * (T*(n + 1) - n)
        coeff = np.sqrt(1 - lmbda**2) * (lmbda**n) * gTn**2
        state[n * dim + n] = coeff
        normalization += np.abs(coeff)**2
    state /= np.sqrt(normalization)
    return state, normalization

def applyNqsbpBisidePcTmsv(lmbda, T, g, dim):
    T = safeT(T)
    state = np.zeros(dim * dim, dtype=complex)
    normalization = 0.0
    for n in range(dim):
        gTn = T**(n - 1) * (T - n * (1 - T))**2
        coeff = np.sqrt(1 - lmbda**2) * (lmbda**n) * gTn
        amplifiedCoeff = coeff * (g**(4 * n)) / ((g**2 + 1)**n)
        state[n * dim + n] = amplifiedCoeff
        normalization += np.abs(amplifiedCoeff)**2
    state /= np.sqrt(normalization)
    return state, normalization

dim = 12
lmbda = 0.5
g = 1.5
tValues = np.linspace(0.0, 0.5, 25)

originalTmsv = tmsvState(lmbda, dim)
probsOriginal = reducedStateModeA(originalTmsv, dim)
coherenceOriginal = coherenceFromProbabilities(probsOriginal)

cohSingle = []
pSingle = []
cohBiside = []
pBiside = []
cohNqsbp = []
pNqsbp = []

for T in tValues:
    sState, sProb = applySingleSideCatalysisTmsv(lmbda, T, dim)
    bState, bProb = applyBisideCatalysisTmsv(lmbda, T, dim)
    nState, nProb = applyNqsbpBisidePcTmsv(lmbda, T, g, dim)

    cohSingle.append(coherenceFromProbabilities(reducedStateModeA(sState, dim)))
    pSingle.append(sProb)

    cohBiside.append(coherenceFromProbabilities(reducedStateModeA(bState, dim)))
    pBiside.append(bProb)

    cohNqsbp.append(coherenceFromProbabilities(reducedStateModeA(nState, dim)))
    pNqsbp.append(nProb)

plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.plot(tValues, [coherenceOriginal]*len(tValues), 'k--', label="Original TMSV")
plt.plot(tValues, cohSingle, '^-r', label="Single-Side PC")
plt.plot(tValues, cohBiside, 's--b', label="Biside PC")
plt.plot(tValues, cohNqsbp, 'd-.', color='purple', label="nQSBP Biside PC-TMSV")
plt.xlabel("Transmittance T")
plt.ylabel("Coherence")
plt.title("Coherence vs Transmittance (λ = 0.5)")
plt.xlim(0, 0.5)
plt.ylim(0, 2.0)
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(tValues, pSingle, '^-r', label="Single-Side PC Success Prob.")
plt.plot(tValues, pBiside, 's--b', label="Biside PC Success Prob.")
plt.plot(tValues, pNqsbp, 'd-.', color='purple', label="nQSBP Biside PC Success Prob.")
plt.xlabel("Transmittance T")
plt.ylabel("Success Probability")
plt.title("Success Probability vs T (λ = 0.5)")
plt.xlim(0, 0.5)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
