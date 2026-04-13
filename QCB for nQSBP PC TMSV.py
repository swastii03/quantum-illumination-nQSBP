!pip install qutip --quiet

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, qeye, ptrace, thermal_dm
from scipy.optimize import minimize_scalar
from scipy.linalg import fractional_matrix_power
from math import factorial

nMax = 16
R = 0.01
Nth = 1.0
T = 0.9
g = 2.0
lambdaVals = np.linspace(0.01, 0.5, 15)
xi = np.arcsin(np.sqrt(R))

a = destroy(nMax)
ad = a.dag()

def thermalState(nTh):
    return thermal_dm(nMax, nTh)

def mixingOperator():
    return (xi * tensor(qeye(nMax), ad, a) - np.conj(xi) * tensor(qeye(nMax), a, ad)).expm()

def createNqsbpSinglePcTmsv(lam):
    coeffs = []
    norm = 0
    for j in range(nMax):
        cj = (
            g**j *
            (1 - T * lam**2)**1.5 *
            lam**j *
            T**((j - 1)/2) *
            (T - j * (1 - T))
        )
        coeffs.append(cj)
        norm += abs(cj)**2
    coeffs = [c / np.sqrt(norm) for c in coeffs]
    state = sum([coeffs[n] * tensor(basis(nMax, n), basis(nMax, n)) for n in range(nMax)])
    return state.unit()

def createNqsbpBisidePcTmsv(lam):
    coeffs = []
    norm = 0
    for n in range(nMax):
        g2n = g**(2 * n)
        factor = (
            np.sqrt(1 - lam**2) *
            lam**n *
            T**(n - 1) *
            (T - n * (1 - T))**2 *
            g2n *
            factorial(n) /
            ((n + 1)**n * (g**2 + 1)**n)
        )
        coeffs.append(factor)
        norm += abs(factor)**2
    coeffs = [c / np.sqrt(norm) for c in coeffs]
    state = sum([coeffs[n] * tensor(basis(nMax, n), basis(nMax, n)) for n in range(nMax)])
    return state.unit()

def calculateRho0Rho1(inputState):
    rhoAB = inputState * inputState.dag()
    rhoA = ptrace(rhoAB, 0)
    rhoC = thermalState(Nth)
    rho0 = tensor(rhoA, rhoC)

    rhoCAdjusted = thermalState(Nth / (1 - R))
    rhoABC = tensor(rhoAB, rhoCAdjusted)
    U = mixingOperator()
    rhoMixed = U * rhoABC * U.dag()
    rho1 = ptrace(rhoMixed, [0, 1])
    return rho0, rho1

def quantumChernoffBound(rho0, rho1):
    rho0Mat = rho0.full()
    rho1Mat = rho1.full()
    def overlapTrace(t):
        try:
            A = fractional_matrix_power(rho0Mat, t)
            B = fractional_matrix_power(rho1Mat, 1 - t)
            return np.real(np.trace(A @ B))
        except:
            return 1.0
    result = minimize_scalar(overlapTrace, bounds=(0, 1), method='bounded')
    return result.fun

qcbSingle, qcbBiside = [], []

print("Calculating QCB values for nQSBP PC-TMSV states:\n")
for lam in lambdaVals:
    state1 = createNqsbpSinglePcTmsv(lam)
    rho0, rho1 = calculateRho0Rho1(state1)
    qcb1 = quantumChernoffBound(rho0, rho1)
    qcbSingle.append(qcb1)
    print(f"λ = {lam:.3f} → QCB (Single-side) = {qcb1:.7f}")

    state2 = createNqsbpBisidePcTmsv(lam)
    rho0b, rho1b = calculateRho0Rho1(state2)
    qcb2 = quantumChernoffBound(rho0b, rho1b)
    qcbBiside.append(qcb2)
    print(f"λ = {lam:.3f} → QCB (Biside)      = {qcb2:.7f}\n")

plt.figure(figsize=(8, 5))
plt.plot(lambdaVals, qcbSingle, 'o-', label='nQSBP Single-side PC-TMSV', color='green')
plt.plot(lambdaVals, qcbBiside, 's-', label='nQSBP Biside PC-TMSV', color='red')
plt.xlabel(r"$\lambda = \tanh^2(r)$", fontsize=14)
plt.ylabel("Quantum Chernoff Bound", fontsize=14)
plt.title("QCB vs λ for nQSBP PC-TMSV States", fontsize=16)
plt.grid(True)
plt.xlim(0, 0.5)
plt.ylim(0.3, 0.8)
plt.legend()
plt.tight_layout()
plt.show()
