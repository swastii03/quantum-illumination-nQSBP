!pip install qutip --quiet

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, qeye, ptrace, thermal_dm
from scipy.optimize import minimize_scalar
from scipy.linalg import fractional_matrix_power

nMax = 16
r = 0.01
nTh = 1.0
lambdaVals = np.linspace(0.01, 0.5, 15)
xi = np.arcsin(np.sqrt(r))
tCatalysis = 0.9

a = destroy(nMax)
ad = a.dag()

def mixingOperator():
    return (xi * tensor(qeye(nMax), ad, a) - np.conj(xi) * tensor(qeye(nMax), a, ad)).expm()

def thermalState(nTh):
    return thermal_dm(nMax, nTh)

def calculateRho0Rho1(state):
    rhoAB = state * state.dag()
    rhoA = ptrace(rhoAB, 0)
    rhoC = thermalState(nTh)
    rho0 = tensor(rhoA, rhoC)

    rhoCAdj = thermalState(nTh / (1 - r))
    rhoABC = tensor(rhoAB, rhoCAdj)
    u = mixingOperator()
    rhoMixed = u * rhoABC * u.dag()
    rho1 = ptrace(rhoMixed, [0, 1])
    return rho0, rho1

def quantumChernoffBound(rho0, rho1):
    rho0Mat = rho0.full()
    rho1Mat = rho1.full()

    def overlapTrace(t):
        try:
            aMat = fractional_matrix_power(rho0Mat, t)
            bMat = fractional_matrix_power(rho1Mat, 1 - t)
            return np.real(np.trace(aMat @ bMat))
        except:
            return 1.0

    result = minimize_scalar(overlapTrace, bounds=(0, 1), method='bounded')
    return result.fun

def createPcTmssSingle(lam, t):
    return sum([
        lam**n * t**((n - 1) / 2) * (t - n * (1 - t)) * tensor(basis(nMax, n), basis(nMax, n))
        for n in range(nMax)
    ]).unit()

def createPcTmssBiside(lam, t):
    return sum([
        lam**n * t**(n - 1) * (t - n * (1 - t))**2 * tensor(basis(nMax, n), basis(nMax, n))
        for n in range(nMax)
    ]).unit()

qcbPcSingle, qcbPcBiside = [], []

for lam in lambdaVals:
    psiSingle = createPcTmssSingle(lam, tCatalysis)
    rho0S, rho1S = calculateRho0Rho1(psiSingle)
    qcbS = quantumChernoffBound(rho0S, rho1S)
    qcbPcSingle.append(qcbS)

    psiBiside = createPcTmssBiside(lam, tCatalysis)
    rho0B, rho1B = calculateRho0Rho1(psiBiside)
    qcbB = quantumChernoffBound(rho0B, rho1B)
    qcbPcBiside.append(qcbB)

    print(f"λ = {lam:.3f} → QCB Single = {qcbS:.6f}, QCB Biside = {qcbB:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(lambdaVals, qcbPcSingle, '--o', label='Single-side PC-TMSV')
plt.plot(lambdaVals, qcbPcBiside, '-.s', label='Biside PC-TMSV')
plt.xlabel(r'$\lambda = \tanh^2(r)$')
plt.ylabel('Quantum Chernoff Bound (QCB)')
plt.title('QCB vs λ for PC-TMSV States')
plt.ylim(0.3, 0.8)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
