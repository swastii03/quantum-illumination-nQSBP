
# Noiseless Linear Amplification of Non-Gaussian States in Quantum Illumination

This project implements a simulation study of quantum illumination (QI) using various non-Gaussian quantum states derived from the Two-Mode Squeezed Vacuum (TMSV) state. It models photon-catalyzed TMSV (PC-TMSV) states and applies n-photon quantum scissors with bunched photons (nQSBP) to amplify them noiselessly. Key performance metrics such as coherence, success probability, Quantum Chernoff Bound (QCB), and entanglement entropy are evaluated.



## Project Structure

1. qcbForTmsv.py
Simulates the Quantum Chernoff Bound (QCB) for the standard Two-Mode Squeezed Vacuum (TMSV) state. This serves as the baseline for quantum hypothesis testing under thermal noise.

2. qcbForPcTmsv.py
Calculates QCB for the photon-catalyzed TMSV (PC-TMSV) states. Both single-side and bi-side catalysis are considered to evaluate their benefits over TMSV.

3. qcbForNqsbpPcTmsv.py
Simulates QCB for nQSBP-amplified PC-TMSV states, providing insight into the effect of noiseless amplification (gain g) on state discrimination performance.

4. qcbForAllStates.py
Provides a consolidated comparison of QCB values for all four categories: TMSV, PC-TMSV (single and bi-side), and nQSBP-enhanced PC-TMSV (single and bi-side).

5. cVsTForCoherentStates.py
Simulates the coherence enhancement for a single-mode coherent state using photon catalysis. Coherence is measured using Shannon entropy of the photon number distribution.

6. cVsTForPcTmsvStates.py
Computes coherence and success probability for single-side and bi-side PC-TMSV states across different transmittance (T) values.

7. cVsTForNqsbpSinglesidePcTmsvStates.py
Simulates the same metrics as above but for nQSBP-amplified single-side PC-TMSV states.

8. cVsTForNqsbpBisidePcTmsvStates.py
Plots coherence and success probability for nQSBP-amplified bi-side PC-TMSV states as a function of transmittance.

9. cVsTForNqsbpSinglesideVsBisidePcTmsvStates.py
Directly compares the performance of nQSBP single-side and bi-side configurations, identifying optimal quantum state engineering regimes.

10. entropyEntanglementPcTmsv.py
Computes entanglement entropy (using von Neumann entropy) of photon-catalyzed TMSV states for varying values of transmittance.

11. entropyEntanglementNqsbpSingleside.py
Calculates entanglement entropy for nQSBP-amplified single-side PC-TMSV states.

12. entropyEntanglementNqsbpBiside.py
Similar to the above, but for nQSBP-amplified bi-side PC-TMSV states.

13. Bhattacharyya Bound for all 4 states.py
Computes and compares the Bhattacharyya Bound for TMSV, PC-TMSV (single-side and biside), and nQSBP-amplified PC-TMSV. Useful for understanding relative performance across all four schemes.

14. Bhattacharyya Bound for biside PC TMSV vs biside nQSBP.py
Compares the Bhattacharyya Bound of biside PC-TMSV with its nQSBP-amplified version. Demonstrates the effect of amplification in biside catalysis.

15. Bhattacharyya Bound for biside PC TMSV.py
Calculates the Bhattacharyya Bound for biside PC-TMSV states over a range of squeezing parameters. Focuses on the performance of dual-mode catalysis without amplification.

16. Bhattacharyya Bound for PC TMSV.py
Evaluates the Bhattacharyya Bound for both single-side and biside PC-TMSV configurations. Establishes the baseline performance for catalyzed states.

17. Bhattacharyya Bound for singleside PC TMSV vs singleside nQSBP.py
Compares the Bhattacharyya Bound between single-side PC-TMSV and its nQSBP-enhanced variant. Captures the effect of probabilistic amplification on detection performance.


## Tech Stack


1. Python 3.10+

2. NumPy – array and numerical operations

3. Matplotlib – plotting graphs

4. SciPy – scientific utilities and matrix operations

5. QuTiP – quantum operator and state simulations
## Usage


To run simulations, use any Python environment (e.g., VS Code, Jupyter Notebook, or terminal). All scripts are standalone .py files and can be executed individually.

## Output & Results

These simulations are used to evaluate the performance of various configurations in quantum illumination scenarios.

Both single-side and bi-side photon catalysis increase the coherence significantly compared to the original TMSV state in PC. The effect is most pronounced at intermediate values of beam splitter transmittance, where the catalytic interaction strength is achieved.

Further, the implementation of nQSBP amplification on PC-TMSV states shows an additional boost in coherence. In particular, the biside photon-catalyzed TMSV state with nQSBP amplification exhibits the highest coherence values across the transmittance range, although it also suffers from a decrease in the success probability of state generation.

The Quantum Chernoff Bound (QCB) is computed for each of the quantum states under thermal noise and beam splitter mixing. The results indicate that both PC-TMSV and nQSBP-PC-TMSV states achieve lower QCB values compared to standard TMSV states, implying better performance in quantum hypothesis testing tasks. 

Additionally, the Bhattacharyya Bound is computed for similar configurations. This measure also confirms that photon catalysis, particularly when combined with nQSBP amplification, results in enhanced distinguishability of the signal and noise hypotheses. 

The plots generated as part of the project include coherence versus transmittance, success probability versus transmittance, and both QCB and Bhattacharyya Bound versus the squeezing parameter lambda. These plots offer a clear visual interpretation of how quantum state engineering techniques can improve performance in noisy quantum sensing environments.

Sample Outputs: To access sample simulation images, navigate through the undermentioned Googe Drive Link. 
https://drive.google.com/drive/folders/1YNKZy7_rUCDmTFISWyEqSe1VqUyCjdrY?usp=sharing
