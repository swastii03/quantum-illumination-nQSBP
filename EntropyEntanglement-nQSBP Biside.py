import numpy as np
import matplotlib.pyplot as plt
import math

def vonNeumannEntropy(cn):
    probs = cn**2
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def nqsbpPcTmsvCoeffs(t1, t2, r, nMax=30, n=3, g=1.5):
    lambdaVal = np.tanh(r)
    a = np.zeros(nMax)
    denomConst = ((n + 1)**(2 * n)) * ((g**2 + 1)**(2 * n))
    factNSq = math.factorial(n)**2
    for j in range(nMax):
        if j == 0:
            t1Term = 1 / (t1**2)
            t2Term = 1 / (t2**2)
        else:
            t1Term = t1**(2 * j - 2)
            t2Term = t2**(2 * j - 2)
        aj = factNSq * (g**(4 * j)) * (lambdaVal**(2 * j))
        aj *= t1Term * t2Term
        aj *= (t1 - j * (1 - t1))**4 * (t2 - j * (1 - t2))**4
        a[j] = aj / denomConst
    norm = np.sqrt(np.sum(a**2))
    if norm == 0:
        norm = 1e-15
    cn = a / norm
    return cn

thetaVals = np.linspace(0, np.pi / 2, 100)
t1Vals = np.cos(thetaVals)**2
t2Vals = np.cos(thetaVals)**2
r = np.arcsinh(np.sqrt(0.05))
g = 0.8
n = 3
nMax = 30
eTmsv = (np.cosh(r)**2 * np.log2(np.cosh(r)**2) - np.sinh(r)**2 * np.log2(np.sinh(r)**2))
eNqsbp = np.zeros((len(t1Vals), len(t2Vals)))

for i, t1 in enumerate(t1Vals):
    for j, t2 in enumerate(t2Vals):
        cnNqsbp = nqsbpPcTmsvCoeffs(t1, t2, r, nMax=nMax, n=n, g=g)
        eNqsbp[i, j] = vonNeumannEntropy(cnNqsbp)

rE = (eNqsbp - eTmsv) / eTmsv

plt.figure(figsize=(7, 5))
contour = plt.contourf(t1Vals, t2Vals, rE.T, levels=50, cmap='RdBu_r')
plt.colorbar(label='$r_E = (E_{nQSBP} - E_{TMSV}) / E_{TMSV}$')
plt.xlabel('$T_1 = \\cos^2 \\theta_1$')
plt.ylabel('$T_2 = \\cos^2 \\theta_2$')
plt.title('Entanglement Enhancement for nQSBP Biside PC-TMSV')
plt.legend()
plt.show()
