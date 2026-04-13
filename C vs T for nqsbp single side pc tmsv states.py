import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    return np.math.factorial(n)

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

def applyNqsbpPcTmsv(lmbda, T, g, dim):
    state = np.zeros(dim * dim, dtype=complex)
    normalization = 0.0
    for n in range(dim):
        if T == 0 and n > 0:
            gTn = 0
        else:
            gTn = T**((n-1)/2) * (T*(n + 1) - n)
        coeff = np.sqrt(1 - lmbda**2) * (lmbda**n) * gTn
        amplifiedCoeff = coeff * (g**(2*n)) / ((g**2 + 1)**(n/2))
        state[n * dim + n] = amplifiedCoeff
        normalization += np.abs(amplifiedCoeff)**2
    state /= np.sqrt(normalization)
    return state, normalization

def reducedStateModeA(state, dim):
    probs = np.zeros(dim)
    for n in range(dim):
        index = n * dim + n
        probs[n] = np.abs(state[index])**2
    return probs

def coherenceFromProbabilities(probs):
    epsilon = 1e-12
    return -np.sum(probs * np.log(probs + epsilon))

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
    stateSingle, ps = applySingleSideCatalysisTmsv(lmbda, T, dim)
    probsSingle = reducedStateModeA(stateSingle, dim)
    cohSingle.append(coherenceFromProbabilities(probsSingle))
    pSingle.append(ps)

    stateBiside, pb = applyBisideCatalysisTmsv(lmbda, T, dim)
    probsBiside = reducedStateModeA(stateBiside, dim)
    cohBiside.append(coherenceFromProbabilities(probsBiside))
    pBiside.append(pb)

    stateNqsbp, pnq = applyNqsbpPcTmsv(lmbda, T, g, dim)
    probsNqsbp = reducedStateModeA(stateNqsbp, dim)
    cohNqsbp.append(coherenceFromProbabilities(probsNqsbp))
    pNqsbp.append(pnq)

plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.plot(tValues, [coherenceOriginal]*len(tValues), 'o-k', label="Original TMSV Coherence")
plt.plot(tValues, cohSingle, '^-r', label="Single-Side PC Coherence")
plt.plot(tValues, cohBiside, 's--b', label="Bi-Side PC Coherence")
plt.plot(tValues, cohNqsbp, 'd-.m', label="n-QSBP Amplified PC-TMSV")
plt.xlabel("Transmittance T")
plt.ylabel("Coherence (Shannon Entropy)")
plt.title("Coherence vs T for TMSV (λ = 0.5)")
plt.xlim(0, 0.5)
plt.ylim(0, 2.0)
plt.xticks(np.arange(0, 0.41, 0.1))
plt.yticks(np.arange(0, 2.1, 0.5))
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tValues, pSingle, '^-r', label="Single-Side PC Success Prob.")
plt.plot(tValues, pBiside, 's--b', label="Bi-Side PC Success Prob.")
plt.plot(tValues, pNqsbp, 'd-.m', label="n-QSBP Success Prob.")
plt.xlabel("Transmittance T")
plt.ylabel("Success Probability")
plt.title("Success Probability vs T for TMSV (λ = 0.5)")
plt.xlim(0, 0.5)
plt.xticks(np.arange(0, 0.41, 0.1))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
