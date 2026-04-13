!pip install qutip --quiet

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

nMax = 15
cutoff = nMax + 1
r = np.arcsinh(np.sqrt(0.05))
nE = 1
tVals = [0.1, 0.2, 0.3]
kVals = np.logspace(0.5, 5.5, 50)

def thermalState(n, cutoff):
    p = [(n / (n + 1))**m / (n + 1) for m in range(cutoff)]
    return Qobj(np.diag(p))

def tmsvKet(lam, cutoff):
    state = sum([(np.sqrt(1 - lam**2) * lam**n) * tensor(basis(cutoff, n), basis(cutoff, n)) for n in range(cutoff)])
    return state.unit()

def applyPcOperator(r, t):
    lam = np.tanh(r)
    psi = tmsvKet(lam, cutoff)
    a1 = tensor(destroy(cutoff), qeye(cutoff))
    a2 = tensor(qeye(cutoff), destroy(cutoff))
    n1 = a1.dag() * a1
    n2 = a2.dag() * a2
    coeff = (1 - t) / t
    tN1 = (np.log(t) / 2.0) * n1
    tN2 = (np.log(t) / 2.0) * n2
    op1 = (coeff * n1 - qeye([cutoff, cutoff])) * tN1.expm()
    op2 = (coeff * n2 - qeye([cutoff, cutoff])) * tN2.expm()
    pcOp = op1 * op2
    newState = (pcOp * psi).unit()
    return newState

def bhattacharyyaBound(rho0, rho1, k):
    sqrtRho0 = rho0.sqrtm()
    fidelityMatrix = (sqrtRho0 * rho1 * sqrtRho0).sqrtm()
    traceVal = fidelityMatrix.tr()
    return 0.5 * (traceVal ** k)

results = {}
rhoTh = thermalState(nE, cutoff)

for t in tVals:
    psiPc = applyPcOperator(r, t)
    rhoPc = ket2dm(psiPc)
    rhoS = rhoPc.ptrace(0)
    rhoI = rhoPc.ptrace(1)
    rho0 = tensor(rhoTh, rhoI)
    rho1 = tensor(rhoS, rhoI)
    bVals = []
    for k in kVals:
        pB = bhattacharyyaBound(rho0, rho1, k)
        bVals.append(np.log10(pB))
    results[t] = bVals

plt.figure(figsize=(8, 5))
for t in tVals:
    plt.plot(np.log10(kVals), results[t], label=f'PC-TMSV T={t}')
plt.xlabel(r'$\log_{10} K$')
plt.ylabel(r'$\log_{10} P^{\mathrm{err}}_B$')
plt.title('Bhattacharyya Bound for PC-TMSV (Replicating Fig. 3a)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
