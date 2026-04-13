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

dim = 12
lmbda = 0.5
g = 1.5
TValues = np.linspace(0.0, 0.5, 25)

originalTmsv = tmsvState(lmbda, dim)
probsOriginal = reducedStateModeA(originalTmsv, dim)
coherenceOriginal = coherenceFromProbabilities(probsOriginal)

cohNqsbpSingle = []
pNqsbpSingle = []
cohNqsbpBiside = []
pNqsbpBiside = []

for T in TValues:
    nStateBi, nProbBi = applyNqsbpBisidePcTmsv(lmbda, T, g, dim)
    nStateSi, nProbSi = applyNqsbpPcTmsv(lmbda, T, g, dim)

    cohNqsbpBiside.append(coherenceFromProbabilities(reducedStateModeA(nStateBi, dim)))
    pNqsbpBiside.append(nProbBi)

    cohNqsbpSingle.append(coherenceFromProbabilities(reducedStateModeA(nStateSi, dim)))
    pNqsbpSingle.append(nProbSi)

plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.plot(TValues, [coherenceOriginal]*len(TValues), 'k--', label="Original TMSV")
plt.plot(TValues, cohNqsbpSingle, 'o-g', label="nQSBP Single-Side PC")
plt.plot(TValues, cohNqsbpBiside, 'd-.', color='purple', label="nQSBP Biside PC")
plt.xlabel("Transmittance T")
plt.ylabel("Coherence")
plt.title("Coherence vs Transmittance (λ = 0.5)")
plt.xlim(0, 0.5)
plt.ylim(0, 2.0)
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(TValues, pNqsbpSingle, 'o-g', label="nQSBP Single-Side PC Success Prob.")
plt.plot(TValues, pNqsbpBiside, 'd-.', color='purple', label="nQSBP Biside PC Success Prob.")
plt.xlabel("Transmittance T")
plt.ylabel("Success Probability")
plt.title("Success Probability vs T (λ = 0.5)")
plt.xlim(0, 0.5)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
