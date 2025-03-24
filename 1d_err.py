from math import sqrt
import numpy as np
from numba import njit, prange
from tqdm import tqdm
import time
import psutil
import os
from tabulate import tabulate

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)

m = 1
w = 1
hcut = 1

N_Cycles = 1000
Time_Step = 0.01
N_Values = np.array([1, 10, 100, 500])
Therm_Steps = 200
Delta = 1e-10
Lambda = 0.0001
block_size = 100

@njit
def WaveFunction(x, alpha):
    return np.exp(-0.5 * alpha**2 * (np.sum(x**2, axis=1)))

@njit
def LocalEnergy(x, alpha, D):
    return 0.5 * ((m * w**2 - hcut**2 * alpha**4 * m**(-1)) * np.sum(np.sum(x**2, axis=1)) + x.shape[0] * D * alpha**2 * hcut**2 * m**(-1))

def RandomMatrix(N_Cycles, N, D):
    rng = np.random.default_rng(41)
    return np.random.normal(0, 1, (N_Cycles, N, D))

@njit
def DriftForce(x: np.ndarray, alpha: float):
    return -alpha**2 * x

@njit
def GF(xOld: np.ndarray, xNew: np.ndarray, F: np.ndarray, alpha):
    return np.exp(-(0.5 * (0.25 * Time_Step * (np.sum(DriftForce(xOld, alpha)**2, axis=1) - np.sum(DriftForce(xNew, alpha)**2, axis=1)) + np.sum((xOld - xNew) * (DriftForce(xNew, alpha) - DriftForce(xOld, alpha)), axis=1))))

@njit
def MonteCarloSampling(Alpha, MCMatrix, xOld, total_steps, rejected_steps, D):
    MCEnergy = np.zeros(N_Cycles - Therm_Steps)
    psiOld = WaveFunction(xOld, Alpha)
    for j in range(1, N_Cycles):
        if j < Therm_Steps:
            xNew = xOld + 0.5 * DriftForce(xOld, Alpha) * Time_Step + MCMatrix[j, :, :] * sqrt(Time_Step)
            psiNew = WaveFunction(xNew, Alpha)
            moves = (((psiNew**2) * GF(xOld, xNew, DriftForce(xOld, Alpha), Alpha)) / ((psiOld**2))) - np.random.normal(0, 1)
            moves = np.where(moves > 0, 1, 0)
            matrixmoves = np.ones((N, D)) * moves[:, np.newaxis]
            xOld[:, :] += (xNew - xOld) * matrixmoves
            psiOld = psiNew
        else:
            xNew = xOld + 0.5 * DriftForce(xOld, Alpha) * Time_Step + MCMatrix[j, :, :] * sqrt(Time_Step)
            psiNew = WaveFunction(xNew, Alpha)
            total_steps += xOld.shape[0]
            moves = (((psiNew**2) * GF(xOld, xNew, DriftForce(xOld, Alpha), Alpha)) / ((psiOld**2))) - np.random.normal(0, 1)
            moves = np.where(moves > 0, 1, 0)
            matrixmoves = np.ones((N, D)) * moves[:, np.newaxis]
            rejected_steps += np.sum(1 - moves)
            if np.all(moves != 1):
                MCEnergy[j - Therm_Steps] = MCEnergy[j - Therm_Steps - 1]
            else:
                xOld += (xNew - xOld) * matrixmoves
                psiOld = WaveFunction(xOld + (xNew - xOld) * matrixmoves, Alpha)
                MCEnergy[j - Therm_Steps] = LocalEnergy(xOld, Alpha, D)

    MeanEnergy = np.sum(MCEnergy) / (N_Cycles - Therm_Steps)
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
    for N in N_Values:
        print(f"Starting gradient descent for N = {N}")
        max_iterations = 400 if N == 1 else 200 if N == 10 else 50 if N == 100 else 40 if N == 500 else 0
        optimal_alphas = []

        for initial_alpha in [0.95, 1.05]:
            total_steps = 0
            rejected_steps = 0
            alpha = initial_alpha
            MCMatrix = RandomMatrix(N_Cycles, N, D)
            xOld = Time_Step * (np.random.normal(0, 1, (N, D)))

            for iteration in tqdm(range(max_iterations), desc=f"N={N}, alpha_initial={initial_alpha}", unit="iteration"):
                energy_up, total_steps, rejected_steps, _ = MonteCarloSampling(alpha + 0.01, MCMatrix, xOld, total_steps, rejected_steps, D)
                energy_down, total_steps, rejected_steps, _ = MonteCarloSampling(alpha - 0.01, MCMatrix, xOld, total_steps, rejected_steps, D)
                gradient = (energy_up - energy_down) / 0.02
                if abs(gradient) < Delta:
                    print(f"Converged after {iteration + 1} iterations.")
                    break
                alpha = alpha - Lambda * gradient
                if alpha < 0.7:
                    alpha = 0.7
                if alpha > 1.3:
                    alpha = 1.3

            optimal_alphas.append(alpha)

        optimal_alpha = np.mean(optimal_alphas)
        MCMatrix = RandomMatrix(N_Cycles, N, D)
        xOld = Time_Step * (np.random.normal(0, 1, (N, D)))
        ground_state_energy, total_steps_ground, rejected_steps_ground, MCEnergy = MonteCarloSampling(optimal_alpha, MCMatrix, xOld, 0, 0, D)
        rejection_rate = (rejected_steps_ground / total_steps_ground) * 100 if total_steps_ground > 0 else 0
        if D=3:
            sigma2, tau_bar = ErrorHandling(MCEnergy)
        else: 
            sigma2=0
            tau_bar=0
            
        err = sqrt(sigma2 * abs(tau_bar) / (N_Cycles - Therm_Steps))

        data_to_print = [optimal_alpha, ground_state_energy, rejection_rate, err]
        headers = [f"OPTIMAL ALPHA FOR N={N}, D={D}", f"GROUND STATE ENERGY FOR N={N}, D={D}", f"REJECTION RATE FOR GROUND STATE ENERGY AT N={N}, D={D} in %", f"ERROR FOR N={N}, D={D}]"]
        print(tabulate([data_to_print], headers=headers, tablefmt="grid"))
end_time = time.time() # Stop counting Running Time
runtime = end_time - start_time
print(f"Total runtime: {runtime:.6f} seconds")

print(f"Memory usage: {get_memory_usage():2f} MB")
