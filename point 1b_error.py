import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit, prange
from tqdm import tqdm
import time

# Constants
m = 1
w = 1
hcut = 1

N_Alpha = 40
N_Cycles = 100000
StepSize = np.array([2, 0.6, 0.2, 0.08], dtype=np.float64)
Nvalues = np.array([1, 10, 100, 500])
Therm_steps = 10000


@njit
def WaveFunction(x, alpha):
    return np.exp(-alpha**2 * np.sum(np.sum(x**2, axis=1)) / 2)

@njit
def LocalEnergy(x, N, alpha, D):
    return 0.5 * ((m * w**2 - hcut**2 * alpha**4 * m**(-1)) * np.sum(np.sum(x**2, axis=1)) + N * D * alpha**2 * hcut**2 * m**(-1))

def RandomMatrix(N_Cycles, N, D):
    return np.random.rand(N_Cycles, N, D)

@njit
def MonteCarloSampling(Alpha, Alpha_Pos, N, N_Pos, MCMatrix, xOld, MCEnergy, rejected_steps, D):
    psiOld = WaveFunction(xOld, Alpha)

    for j in range(1, N_Cycles):
        if j < Therm_steps:
            xNew = xOld + StepSize[N_Pos] * (MCMatrix[j, :, :] - 0.5)
            psiNew = WaveFunction(xNew, Alpha)
            if psiOld > 0 and (psiNew ** 2 / psiOld ** 2) > np.random.rand():
                xOld[:, :] = xNew
                psiOld = psiNew            
        else:
            xNew = xOld + StepSize[N_Pos] * (MCMatrix[j, :, :] - 0.5)
            psiNew = WaveFunction(xNew, Alpha)
            if psiOld > 0 and (psiNew ** 2 / psiOld ** 2) > np.random.rand():
                xOld[:, :] = xNew
                psiOld = psiNew
                MCEnergy[Alpha_Pos, j - Therm_steps] = LocalEnergy(xOld, N, Alpha, D)
            else:
                rejected_steps += 1
                MCEnergy[Alpha_Pos, j-Therm_steps] = MCEnergy[Alpha_Pos, j - Therm_steps- 1]

    MeanEnergy = np.sum(MCEnergy, axis=1) / (N_Cycles - Therm_steps)

    return MeanEnergy, rejected_steps
@njit(parallel=True)
def ErrorHandling(MCEnergy_Alpha, block_size=10000):
    MeanEnergy = np.sum(MCEnergy_Alpha) / (N_Cycles - Therm_steps)
    sigma2 = np.sum((MCEnergy_Alpha - MeanEnergy) ** 2) / (N_Cycles - Therm_steps)
    
    numerator = 0.0
    n = MCEnergy_Alpha.shape[0]
    
    num_blocks = n // block_size   #+ (n % block_size != 0)  # Numero totale di blocchi
    
    if not isinstance(num_blocks,int):
        print("Number of MC Cycles is not a multiple of 10000: the number of blocks is not an integer")
        exit()
        
    for block in prange(num_blocks):  # Parallelizzazione con prange
        start = block * block_size
        end = min((block + 1) * block_size, n)
        
        for tau in range(start, end):  # Elaborazione solo sui dati del blocco corrente
            sum_corr = 0.0
            for i in range(n - tau):
                sum_corr += MCEnergy_Alpha[i] * MCEnergy_Alpha[i + tau]
            
            sum_corr /= (n - tau)
            numerator += sum_corr - MeanEnergy**2
    
    numerator /= sigma2  # Normalizzazione finale
    
    return sigma2, numerator


# --- Main Execution ---
start_time = time.time()

for D in [3]:
    print(f"Computing Energies in {D} dimensions...")

    for N_Pos, N in enumerate(Nvalues):
        rejected_steps = 0
        alpha_values = np.linspace(0.7, 1.3, N_Alpha)
        MCEnergy = np.zeros((N_Alpha, N_Cycles-Therm_steps))
        sigma2=np.zeros(N_Alpha)
        tau_bar=np.zeros(N_Alpha)
        
        for Alpha_Pos in tqdm(range(N_Alpha)):
            Alpha = alpha_values[Alpha_Pos]
            xOld = StepSize[N_Pos] * (np.random.rand(N, D) - 0.5)
            psiOld = WaveFunction(xOld, Alpha)
            MCEnergy[Alpha_Pos, 0] = LocalEnergy(xOld, N, Alpha, D)
            MCMatrix = RandomMatrix(N_Cycles, N, D)

            MeanEnergy, rejected_steps = MonteCarloSampling(
                Alpha, Alpha_Pos, N, N_Pos, MCMatrix, xOld, MCEnergy, rejected_steps, D
            )
            sigma2[Alpha_Pos], tau_bar[Alpha_Pos] = ErrorHandling(MCEnergy[Alpha_Pos,:])

        # Calcola total_moves utilizzando np.int64
        total_moves = np.int64(N_Cycles - Therm_steps) * np.int64(N_Alpha)
        rejection_percentage = (rejected_steps / total_moves) * 100
        print(f"Rejection Percentage for N={N}: {rejection_percentage:.2f}%")

        min_index = np.argmin(MeanEnergy)
        best_alpha = alpha_values[min_index]
        min_energy = MeanEnergy[min_index]
        err = np.sqrt(sigma2 * np.abs(tau_bar) / (N_Cycles - Therm_steps))
        min_error = err[min_index]
        print(f"Optimal alpha for N={N}: {best_alpha}, Minimum energy: {min_energy}")
        print(f"Error for minimum energy (Alpha = {best_alpha}): {min_error}")
        plt.figure()
        plt.errorbar(alpha_values, MeanEnergy, yerr=err ,label=f"N={N}", color='b', marker='o')
        plt.xlabel("Alpha values")
        plt.ylabel("Energy")
        plt.title(f"Energy vs Alpha for N={N}, D={D}")
        plt.legend()
        plt.show()
        print(f"Energies in {D} dimensions for {N} harmonic oscillators")

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.6f} seconds")
