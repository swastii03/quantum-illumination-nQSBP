import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, qeye, ptrace, thermal_dm
from scipy.optimize import minimize_scalar
from scipy.linalg import fractional_matrix_power

nMax = 16
r = 0.01
nth = 1.0
lambdaVals = np.linspace(0.01, 0.5, 15)
xi = np.arcsin(np.sqrt(r))

a = destroy(nMax)
ad = a.dag()

def createTmss(lam):
    state = sum([lam**n * tensor(basis(nMax, n), basis(nMax, n)) for n in range(nMax)])
    return state.unit()

def thermalState(nTh):
    return thermal_dm(nMax, nTh)

def mixingOperator():
    return (xi * tensor(qeye(nMax), ad, a) - np.conj(xi) * tensor(qeye(nMax), a, ad)).expm()

def calculateRho0Rho1(tmssState):
    rhoAB = tmssState * tmssState.dag()
    rhoA = ptrace(rhoAB, 0)
    rhoC = thermalState(nth)
    rho0 = tensor(rhoA, rhoC)

    rhoCAdjusted = thermalState(nth / (1 - r))
    rhoABC = tensor(rhoAB, rhoCAdjusted)
    u = mixingOperator()
    rhoMixed = u * rhoABC * u.dag()
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
        except Exception as e:
            print(f"Error at t = {t}: {e}")
            return 1.0

    try:
        result = minimize_scalar(overlapTrace, bounds=(0, 1), method='bounded')
        return result.fun
    except:
        return 1.0

qcbVals = []
for lam in lambdaVals:
    tmss = createTmss(lam)
    rho0, rho1 = calculateRho0Rho1(tmss)
    qcb = quantumChernoffBound(rho0, rho1)
    qcbVals.append(qcb)
    print(f"λ = {lam:.3f} → QCB = {qcb:.7f}")

plt.figure(figsize=(8, 5))
plt.plot(lambdaVals, qcbVals, label='TMSS (Fock)', color='blue', linewidth=2)
plt.xlabel(r"$\lambda = \tanh^2(r)$", fontsize=14)
plt.ylabel("Quantum Chernoff Bound", fontsize=14)
plt.title("QCB vs λ for TMSS (Exact Fock Simulation)", fontsize=16)
plt.grid(True)
plt.xlim(0.0, 0.5)
plt.legend()
plt.tight_layout()
plt.show()
