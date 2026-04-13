import matplotlib.pyplot as plt

lambdas = [
    0.010, 0.045, 0.080, 0.115, 0.150, 0.185, 0.220, 0.255,
    0.290, 0.325, 0.360, 0.395, 0.430, 0.465, 0.500
]

qcbTmsv = [
    0.7257286, 0.7236786, 0.7190944, 0.7120757, 0.7027290,
    0.6911635, 0.6774866, 0.6618020, 0.6442063, 0.6247882,
    0.6036271, 0.5807918, 0.5563404, 0.5303201, 0.5027669
]

qcbNqsbpSingle = [
    0.725760, 0.724299, 0.721023, 0.715992, 0.709271,
    0.700929, 0.691035, 0.679659, 0.666871, 0.652737,
    0.637324, 0.620694, 0.602908, 0.584025, 0.564103
]

qcbNqsbpBiside = [
    0.725782, 0.724741, 0.722402, 0.718800, 0.713975,
    0.707967, 0.700822, 0.692582, 0.683293, 0.673001,
    0.661750, 0.649585, 0.636550, 0.622689, 0.608045
]

qcbPcSingle = [
    0.7255306, 0.7197618, 0.7070501, 0.6879326, 0.6629667,
    0.6326937, 0.5976220, 0.5582287, 0.5149802, 0.4683540,
    0.4188039, 0.3665244, 0.3108513, 0.2492063, 0.1758748
]

qcbPcBiside = [
    0.7258341, 0.7257922, 0.7256970, 0.7255488, 0.7253479,
    0.7250946, 0.7247891, 0.7244317, 0.7240228, 0.7235625,
    0.7230514, 0.7224896, 0.7218775, 0.7212155, 0.7205040
]

plt.figure(figsize=(10, 6))
plt.plot(lambdas, qcbTmsv, 'x-k', label='TMSV')
plt.plot(lambdas, qcbPcSingle, 'o--', label='PC-TMSV Single-side')
plt.plot(lambdas, qcbPcBiside, 's--', label='PC-TMSV Biside')
plt.plot(lambdas, qcbNqsbpSingle, 'o-', label='nQSBP Single-side PC-TMSV')
plt.plot(lambdas, qcbNqsbpBiside, 's-', label='nQSBP Biside PC-TMSV')

plt.xlabel("λ (squeezing parameter)", fontsize=12)
plt.ylabel("Quantum Chernoff Bound (QCB)", fontsize=12)
plt.title("QCB vs λ for TMSV, PC-TMSV and nQSBP-Amplified States (T=0.5, g=1.5)", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
