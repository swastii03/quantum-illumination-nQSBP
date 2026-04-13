import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

def vonNeumannEntropy(cn):
    probs = cn**2
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def nqsbpPcTmsvCoeffs(t1, t2, r, g=1.2, nMax=30):
    lambdaSq = np.sinh(r)**2 / np.cosh(r)**2
    cn = np.zeros(nMax)

    normFactor = (1 - t1 * lambdaSq)**(3/2) * (1 - t2 * lambdaSq)**(3/2)
    mainTerm = lambdaSq**2 * (1 + t1**2) + t1 * (1 - 4 * lambdaSq + lambdaSq**2)
    mainTerm *= lambdaSq**2 * (1 + t2**2) + t2 * (1 - 4 * lambdaSq + lambdaSq**2)

    for n in range(nMax):
        bracketTerm = (t1 - n * (1 - t1)) * (t2 - n * (1 - t2))
        cn[n] = (g**(4*n)) * (lambdaSq**(2*n)) * bracketTerm / normFactor

    cn /= np.linalg.norm(cn)
    return cn

def pcTmsvCoeffs(t1, t2, r, nMax=30):
    lambdaSq = np.sinh(r)**2 / np.cosh(r)**2
    cn = np.zeros(nMax)

    for n in range(nMax):
        normFactor = t1**(n-1) * t2**(n-1)
        bracketTerm = (t1**2 - n*(1 - t1**2)) * (t2**2 - n*(1 - t2**2))
        cn[n] = lambdaSq**n * normFactor * bracketTerm

    cn /= np.linalg.norm(cn)
    return cn

thetaVals = np.linspace(0, np.pi/2, 100)
t1Vals = np.cos(thetaVals)**2
t2Vals = np.cos(thetaVals)**2

r = np.arcsinh(np.sqrt(0.05))

eTmsv = vonNeumannEntropy(pcTmsvCoeffs(1, 1, r))
ePc = np.zeros((len(t1Vals), len(t2Vals)))
eNqsbpPc = np.zeros((len(t1Vals), len(t2Vals)))

for i, t1 in enumerate(t1Vals):
    for j, t2 in enumerate(t2Vals):
        cnPc = pcTmsvCoeffs(t1, t2, r)
        cnNqsbpPc = nqsbpPcTmsvCoeffs(t1, t2, r)
        ePc[i, j] = vonNeumannEntropy(cnPc)
        eNqsbpPc[i, j] = vonNeumannEntropy(cnNqsbpPc)

rE = (eNqsbpPc - eTmsv) / eTmsv

plt.figure(figsize=(7, 5))
plt.contourf(t1Vals, t2Vals, rE.T, levels=50, cmap='RdBu_r')
plt.colorbar(label='$r_E = (E_{nQSBP-PC} - E_{TMSV}) / E_{TMSV}$')
plt.xlabel('$T_1 = \cos^2 \theta_1$')
plt.ylabel('$T_2 = \cos^2 \theta_2$')
plt.title('Entanglement Enhancement for nQSBP-Single side PC-TMSV')
plt.show()
