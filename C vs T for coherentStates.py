import numpy as np
import matplotlib.pyplot as plt

def factorialValue(n):
    return np.math.factorial(n)

def photonCatalysisCoefficient(tVal, n):
    return tVal**((n - 1) / 2) * (tVal * (n + 1) - n)

def probabilityPhotonCatalysis(alphaVal, tVal):
    return np.exp((tVal - 1) * alphaVal**2) * (
        tVal**3 * alphaVal**4
        + tVal**2 * alphaVal**2 * (3 - 2 * alphaVal**2)
        + tVal * (1 - 4 * alphaVal**2 + alphaVal**4)
        + alphaVal**2
    )

def catalyzedState(alphaVal, tVal, dim=20):
    pCatalysis = probabilityPhotonCatalysis(alphaVal, tVal)
    stateVec = np.zeros(dim, dtype=complex)
    for n in range(dim):
        coeff = (
            np.exp(-alphaVal**2 / 2)
            * alphaVal**n
            * photonCatalysisCoefficient(tVal, n)
            / (np.sqrt(pCatalysis) * factorialValue(n))
        )
        stateVec[n] = coeff
    stateVec /= np.linalg.norm(stateVec)
    return stateVec

def coherenceMeasure(stateVec):
    probs = np.abs(stateVec)**2
    return -np.sum(probs * np.log(probs + 1e-10))

def originalCoherentState(alphaVal, dim=20):
    stateVec = np.array(
        [
            np.exp(-alphaVal**2 / 2) * alphaVal**n / np.sqrt(factorialValue(n))
            for n in range(dim)
        ]
    )
    return stateVec / np.linalg.norm(stateVec)

alphaVal = 0.5
tValues = np.linspace(0.0, 0.435, 10)
originalState = originalCoherentState(alphaVal)
originalCoherence = coherenceMeasure(originalState)

catalyzedCoherences = []
for tVal in tValues:
    catalyzedStateVec = catalyzedState(alphaVal, tVal)
    catalyzedCoherences.append(coherenceMeasure(catalyzedStateVec))

plt.figure(figsize=(8, 5))
plt.plot(tValues, [originalCoherence] * len(tValues), 'o-k', label="$C(|\\alpha\\rangle)$")
plt.plot(tValues, catalyzedCoherences, '*-r', label="$C(|\\psi'\\rangle)$")
plt.xlabel("$T$")
plt.ylabel("Coherence $C$")
plt.title("Coherence Enhancement with Photon Catalysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
