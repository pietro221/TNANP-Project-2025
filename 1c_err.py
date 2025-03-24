from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from tqdm import tqdm
import time
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)

m = 1
w = 1
hcut = 1

N_Alpha = 40
N_Cycles = 100000
timestep = 0.01
Nvalues = np.array([1, 10, 100, 500])
Therm_steps = 20000
sigma_gaussian = 1
block_size = 10000  # Dimensione del blocco

@njit
def WaveFunction(x, alpha):
    return np.exp(-0.5 * alpha**2 * (np.sum(x**2, axis=1)))

@njit
def LocalEnergy(x, alpha, D):
    return 0.5 * ((m * w**2 - hcut**2 * alpha**4 * m**(-1)) * np.sum(np.sum(x**2, axis=1)) + x.shape[0] * D * alpha**2 * hcut**2 * m**(-1))

def RandomMatrix(N_Cycles, N, D):
    rng = np.random.default_rng(41)
    return np.random.normal(0, sigma_gaussian, (N_Cycles, N, D))

@njit
def DriftForce(x: np.ndarray, alpha: float):
    return -alpha**2 * x

@njit
def GF(xOld: np.ndarray, xNew: np.ndarray, F: np.ndarray, alpha):
    return np.exp(-(0.5 * (0.25 * timestep * (np.sum(DriftForce(xOld, alpha)**2, axis=1) - np.sum(DriftForce(xNew, alpha)**2, axis=1)) + np.sum((xOld - xNew) * (DriftForce(xNew, alpha) - DriftForce(xOld, alpha)), axis=1))))

@njit
def MonteCarloSampling(Alpha, Alpha_Pos, N_Pos, MCMatrix, xOld, MCEnergy, total_steps, rejected_steps, D):
    psiOld = WaveFunction(xOld, Alpha)

    for j in range(1, N_Cycles):
        if j < Therm_steps:
            xNew = xOld + 0.5 * DriftForce(xOld, Alpha) * timestep + MCMatrix[j, :, :] * sqrt(timestep)
            psiNew = WaveFunction(xNew, Alpha)
            moves = (((psiNew**2) * GF(xOld, xNew, DriftForce(xOld, Alpha), Alpha)) / ((psiOld**2))) - np.random.normal(0, sigma_gaussian)
            moves = np.where(moves > 0, 1, 0)
            matrixmoves = np.ones((xOld.shape[0], D)) * moves[:, np.newaxis]
            xOld[:, :] += (xNew - xOld) * matrixmoves
            psiOld = psiNew
        else:
            xNew = xOld + 0.5 * DriftForce(xOld, Alpha) * timestep + MCMatrix[j, :, :] * sqrt(timestep)
            psiNew = WaveFunction(xNew, Alpha)
            total_steps += xOld.shape[0]
            moves = (((psiNew**2) * GF(xOld, xNew, DriftForce(xOld, Alpha), Alpha)) / ((psiOld**2))) - np.random.normal(0, sigma_gaussian)
            moves = np.where(moves > 0, 1, 0)
            matrixmoves = np.ones((xOld.shape[0], D)) * moves[:, np.newaxis]
            rejected_steps += np.sum(1 - moves)
            if np.all(moves != 1):
                MCEnergy[Alpha_Pos, j - Therm_steps] = MCEnergy[Alpha_Pos, j - Therm_steps - 1]
            else:
                xOld += (xNew - xOld) * matrixmoves
                psiOld = WaveFunction(xOld, Alpha)
                MCEnergy[Alpha_Pos, j - Therm_steps] = LocalEnergy(xOld, Alpha, D)

    MeanEnergy = np.sum(MCEnergy, axis=1) / (N_Cycles - Therm_steps)
    return MeanEnergy, total_steps, rejected_steps, MCEnergy

@njit(parallel=True)
def ErrorHandling(MCEnergy_Alpha):
    MeanEnergy = np.sum(MCEnergy_Alpha) / MCEnergy_Alpha.shape[0]
    sigma2 = np.sum((MCEnergy_Alpha - MeanEnergy) ** 2) / MCEnergy_Alpha.shape[0]

    numerator = 0.0
    n = MCEnergy_Alpha.shape[0]

    num_blocks = (n + block_size - 1) // block_size
    if not isinstance(num_blocks,int):
        print("Number of MC Cycles is not a multiple of 10000: the number of blocks is not an integer")
        exit()

    for block in prange(num_blocks):
        start = block * block_size
        end = min((block + 1) * block_size, n)

        for tau in range(start, end):
            sum_corr = 0.0
            for i in range(n - tau):
                sum_corr += MCEnergy_Alpha[i] * MCEnergy_Alpha[i + tau]

            sum_corr /= (n - tau)
            numerator += sum_corr - MeanEnergy**2

    numerator /= sigma2

    return sigma2, numerator

start_time = time.time()
for D in [1, 2, 3]:
    print(f"Computing Energies in {D} dimensions...")
    for N_Pos, N in enumerate(Nvalues):
        total_steps = 0
        rejected_steps = 0
        alpha_values = np.linspace(0.7, 1.3, N_Alpha)
        MCEnergy = np.zeros((N_Alpha, N_Cycles - Therm_steps))
        sigma2 = np.zeros(N_Alpha)
        tau_bar = np.zeros(N_Alpha)

        for Alpha_Pos in tqdm(range(N_Alpha)):
            Alpha = alpha_values[Alpha_Pos]
            xOld = timestep * (np.random.normal(0, sigma_gaussian, (N, D)))
            psiOld = WaveFunction(xOld, Alpha)
            MCEnergy[Alpha_Pos, 0] = LocalEnergy(xOld, Alpha, D)
            MCMatrix = RandomMatrix(N_Cycles, N, D)
            MeanEnergy, total_steps, rejected_steps, MatrixEnergy = MonteCarloSampling(Alpha, Alpha_Pos, N_Pos, MCMatrix, xOld, MCEnergy, total_steps, rejected_steps, D)
            MCEnergy = MatrixEnergy
            sigma2[Alpha_Pos], tau_bar[Alpha_Pos] = ErrorHandling(MCEnergy[Alpha_Pos, :])

        rejection_percentage = (rejected_steps / total_steps) * 100
        print(f"Rejection Percentage for N={N}: {rejection_percentage:.2f}%")

        min_index = np.argmin(MeanEnergy)
        best_alpha = alpha_values[min_index]
        min_energy = MeanEnergy[min_index]       
        err = np.sqrt(sigma2 * np.abs(tau_bar) / (N_Cycles - Therm_steps))
        min_error = err[min_index]
        print(f"Optimal alpha for N={N}: {best_alpha}, Minimum energy: {min_energy}")
        print(f"Error for minimum energy (Alpha = {best_alpha}): {min_error}")


        plt.figure()
        plt.errorbar(alpha_values, MeanEnergy, yerr=err, label=f"N={N}", color='b', marker='o')
        plt.xlabel("Alpha values")
        plt.ylabel("Energy")
        plt.title(f"Energy vs Alpha for N={N}, D={D}")
        plt.legend()
        plt.show()
        print(f"Energies in {D} dimensions for {N} harmonic oscillators")

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.6f} seconds")

print(f"Memory usage: {get_memory_usage():.2f} MB")
