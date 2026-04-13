import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.linalg import sqrtm
from numpy.linalg import norm as vecNorm

dim = 16
lambdaVal = 0.25
g = 1.7
ne = 1
kappa = 0.01
t = 0.3
kVals = np.logspace(2, 4, 25, dtype=int)

def vectorToDensity(psi):
    return np.outer(psi, np.conj(psi))

def partialTraceSignal(rho, dim):
    return np.einsum('ijik->jk', rho.reshape(dim, dim, dim, dim))

def thermalState(ne, dim):
    p = [(ne / (ne + 1))**n / (ne + 1) for n in range(dim)]
    return np.diag(p)

def rho0Vec(psi, ne, dim):
    rhoSignal = partialTraceSignal(vectorToDensity(psi), dim)
    return np.kron(rhoSignal, thermalState(ne, dim))

def rho1Vec(psi, ne, kappa, dim):
    rhoSignal = partialTraceSignal(vectorToDensity(psi), dim)
    return np.kron((1 - kappa) * rhoSignal, rhoSignal)

def bhattacharyyaBound(rho0, rho1, k):
    sqrtR0 = sqrtm(rho0)
    product = sqrtR0 @ rho1 @ sqrtR0
    fidelity = np.trace(sqrtm(product)).real
    return 0.5 * (fidelity ** k)

def tmsvState(lambdaVal, dim):
    state = np.array([np.sqrt(1 - lambdaVal**2) * lambdaVal**n for n in range(dim)])
    return np.kron(state, state) / vecNorm(np.kron(state, state))

def pcSinglesideState(lambdaVal, t, dim):
    normFactor = lambdaVal**2 * (1 + t**2) + t * (1 - 4 * lambdaVal**2 + lambdaVal**4)
    const = np.sqrt((1 - t * lambdaVal**2)**3)
    state = np.zeros(dim)
    for n in range(dim):
        f = t**((n - 1) / 2) * (t - n * (1 - t)) if n > 0 else 0.0
        coeff = const * (lambdaVal ** n) * f * normFactor
        state[n] = coeff
    return np.kron(state, state) / vecNorm(np.kron(state, state))

def pcBisideState(lambdaVal, t, dim):
    state = np.zeros(dim)
    for n in range(dim):
        f = t**((n - 1)/2) * (t*(n+1) - n) if not (t == 0 and n == 0) else 0.0
        coeff = np.sqrt(1 - lambdaVal**2) * (lambdaVal ** n) * f
        state[n] = coeff
    return np.kron(state, state) / vecNorm(np.kron(state, state))

def nqsbpOperator(g, dim):
    nCut = dim - 1
    pref = np.sqrt(factorial(nCut) / ((nCut + 1)**nCut * (g**2 + 1)**nCut))
    return np.diag([(g**n) * pref for n in range(dim)])

def nqsbpBisideAmplify(psi, g, dim):
    G = nqsbpOperator(g, dim)
    GG = np.kron(G, G)
    amplifiedPsi = GG @ psi
    return amplifiedPsi / vecNorm(amplifiedPsi)

results = {}

psiTmsv = tmsvState(lambdaVal, dim)
rho0Tmsv = rho0Vec(psiTmsv, ne, dim)
rho1Tmsv = rho1Vec(psiTmsv, ne, kappa, dim)
results["TMSV"] = [bhattacharyyaBound(rho0Tmsv, rho1Tmsv, k) for k in kVals]

psiPcS = pcSinglesideState(lambdaVal, t, dim)
rho0S = rho0Vec(psiPcS, ne, dim)
rho1S = rho1Vec(psiPcS, ne, kappa, dim)
results["PC-Single"] = [bhattacharyyaBound(rho0S, rho1S, k) for k in kVals]

psiPcB = pcBisideState(lambdaVal, t, dim)
rho0B = rho0Vec(psiPcB, ne, dim)
rho1B = rho1Vec(psiPcB, ne, kappa, dim)
results["PC-Biside"] = [bhattacharyyaBound(rho0B, rho1B, k) for k in kVals]

psiAmpS = nqsbpBisideAmplify(psiPcS, g, dim)
rho0As = rho0Vec(psiAmpS, ne, dim)
rho1As = rho1Vec(psiAmpS, ne, kappa, dim)
results["nQSBP-PC-Single"] = [bhattacharyyaBound(rho0As, rho1As, k) for k in kVals]

psiAmpB = nqsbpBisideAmplify(psiPcB, g, dim)
rho0Ab = rho0Vec(psiAmpB, ne, dim)
rho1Ab = rho1Vec(psiAmpB, ne, kappa, dim)
results["nQSBP-PC-Biside"] = [bhattacharyyaBound(rho0Ab, rho1Ab, k) for k in kVals]

plt.figure(figsize=(10, 6))
plt.plot(np.log10(kVals), np.log10(results["TMSV"]), 'k--', label='TMSV')
plt.plot(np.log10(kVals), np.log10(results["PC-Single"]), 'g--', label='PC-TMSV (Single-side)')
plt.plot(np.log10(kVals), np.log10(results["PC-Biside"]), 'b--', label='PC-TMSV (Bi-side)')
plt.plot(np.log10(kVals), np.log10(results["nQSBP-PC-Single"]), 'y-', label='nQSBP Single-side PC-TMSV')
plt.plot(np.log10(kVals), np.log10(results["nQSBP-PC-Biside"]), 'r-', label='nQSBP Bi-side PC-TMSV')
plt.xlabel("log₁₀ K")
plt.ylabel("log₁₀ P_B")
plt.title("Bhattacharyya Bound vs K (T = 0.3, dim = 16)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
