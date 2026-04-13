import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from numpy.linalg import norm as vecNorm
from scipy.linalg import sqrtm

dim = 12
lmbda = 0.5
kappa = 0.01
nE = 1
tVals = [0.1, 0.2, 0.3]
kVals = np.logspace(2, 6, 30, dtype=int)

def vectorToDensity(psi):
    return np.outer(psi, np.conj(psi))

def partialTraceSignal(rho, dim):
    return np.einsum('ijik->jk', rho.reshape(dim, dim, dim, dim))

def thermalState(nE, dim):
    p = [(nE / (nE + 1))**n / (nE + 1) for n in range(dim)]
    return np.diag(p)

def rho0Vec(psi, nE, dim):
    rhoSignal = partialTraceSignal(vectorToDensity(psi), dim)
    return np.kron(rhoSignal, thermalState(nE, dim))

def rho1Vec(psi, nE, kappa, dim):
    rhoSignal = partialTraceSignal(vectorToDensity(psi), dim)
    return np.kron((1 - kappa) * rhoSignal, rhoSignal)

def bhattacharyyaBound(rho0, rho1, k):
    sqrtR0 = sqrtm(rho0)
    fidelity = np.trace(sqrtm(sqrtR0 @ rho1 @ sqrtR0)).real
    return 0.5 * (fidelity ** k)

def tmsvState(lmbda, dim):
    state = np.array([np.sqrt(1 - lmbda**2) * lmbda**n for n in range(dim)])
    return np.kron(state, state) / vecNorm(np.kron(state, state))

def pcBisideState(lmbda, t, dim):
    state = np.zeros(dim)
    for n in range(dim):
        f = t**((n - 1)/2) * (t * (n + 1) - n) if not (t == 0 and n == 0) else 0.0
        coeff = np.sqrt(1 - lmbda**2) * (lmbda ** n) * f
        state[n] = coeff
    return np.kron(state, state) / vecNorm(np.kron(state, state))

boundsTmsv = []
boundsPc = {t: [] for t in tVals}

psiTmsv = tmsvState(lmbda, dim)
rho0T = rho0Vec(psiTmsv, nE, dim)
rho1T = rho1Vec(psiTmsv, nE, kappa, dim)
boundsTmsv = [bhattacharyyaBound(rho0T, rho1T, k) for k in kVals]

for t in tVals:
    psiPc = pcBisideState(lmbda, t, dim)
    rho0 = rho0Vec(psiPc, nE, dim)
    rho1 = rho1Vec(psiPc, nE, kappa, dim)
    boundsPc[t] = [bhattacharyyaBound(rho0, rho1, k) for k in kVals]

plt.figure(figsize=(8, 5))
plt.plot(np.log10(kVals), np.log10(boundsTmsv), 'k--', label='TMSV')
for t in tVals:
    plt.plot(np.log10(kVals), np.log10(boundsPc[t]), label=f'Bi-side PC-TMSV (T={t})')
plt.xlabel("log₁₀ K")
plt.ylabel("log₁₀ P_B")
plt.title("Bhattacharyya Bound vs K — Bi-side PC-TMSV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
