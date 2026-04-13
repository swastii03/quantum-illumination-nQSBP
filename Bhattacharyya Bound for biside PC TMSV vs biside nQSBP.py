import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.linalg import sqrtm
from numpy.linalg import norm as vecnorm

dim = 16
lambdaVal = 0.25
g = 1.7
ne = 1
kappa = 0.01
tVals = [0.1, 0.2, 0.3]
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
    sqrtRho0 = sqrtm(rho0)
    product = sqrtRho0 @ rho1 @ sqrtRho0
    fidelity = np.trace(sqrtm(product)).real
    return 0.5 * (fidelity ** k)

def tmsvState(lambdaVal, dim):
    state = np.array([np.sqrt(1 - lambdaVal**2) * lambdaVal**n for n in range(dim)])
    return np.kron(state, state) / vecnorm(np.kron(state, state))

def pcBisideState(lambdaVal, t, dim):
    state = np.zeros(dim)
    for n in range(dim):
        f = t**((n - 1)/2) * (t*(n+1) - n) if not (t == 0 and n == 0) else 0.0
        coeff = np.sqrt(1 - lambdaVal**2) * (lambdaVal ** n) * f
        state[n] = coeff
    return np.kron(state, state) / vecnorm(np.kron(state, state))

def nqsbpOperator(g, dim):
    nCut = dim - 1
    pref = np.sqrt(factorial(nCut) / ((nCut + 1) ** nCut * (g ** 2 + 1) ** nCut))
    return np.diag([(g ** n) * pref for n in range(dim)])

def nqsbpBisideAmplify(psi, g, dim):
    gOp = nqsbpOperator(g, dim)
    gOpKron = np.kron(gOp, gOp)
    amplifiedPsi = gOpKron @ psi
    return amplifiedPsi / vecnorm(amplifiedPsi)

results = {"TMSV": [], "PCBiside": {}, "nQSBPPCBiside": {}}

psiTmsv = tmsvState(lambdaVal, dim)
rho0Tmsv = rho0Vec(psiTmsv, ne, dim)
rho1Tmsv = rho1Vec(psiTmsv, ne, kappa, dim)
results["TMSV"] = [bhattacharyyaBound(rho0Tmsv, rho1Tmsv, k) for k in kVals]

for t in tVals:
    psiPc = pcBisideState(lambdaVal, t, dim)
    rho0Pc = rho0Vec(psiPc, ne, dim)
    rho1Pc = rho1Vec(psiPc, ne, kappa, dim)
    results["PCBiside"][t] = [bhattacharyyaBound(rho0Pc, rho1Pc, k) for k in kVals]

    psiAmp = nqsbpBisideAmplify(psiPc, g, dim)
    rho0Amp = rho0Vec(psiAmp, ne, dim)
    rho1Amp = rho1Vec(psiAmp, ne, kappa, dim)
    results["nQSBPPCBiside"][t] = [bhattacharyyaBound(rho0Amp, rho1Amp, k) for k in kVals]

plt.figure(figsize=(9, 6))
plt.plot(np.log10(kVals), np.log10(results["TMSV"]), 'k--', label='TMSV')

for t in tVals:
    plt.plot(np.log10(kVals), np.log10(results["PCBiside"][t]), '--', label=f'PCBiside T={t}')
    plt.plot(np.log10(kVals), np.log10(results["nQSBPPCBiside"][t]), '-', label=f'nQSBP PCBiside T={t}')

plt.xlabel("log₁₀ K")
plt.ylabel("log₁₀ P_B")
plt.title("Bhattacharyya Bound vs K (λ = 0.25)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
