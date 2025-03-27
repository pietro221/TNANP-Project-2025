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

# We use natural Units
m = 1
w = 1
hcut = 1

N_Alpha = 40 # N° of alpha points
N_Cycles = 100000 # N° of MC cycles
Time_Step = 0.01 # Same Timestep for each Harmonic Oscillator Number
N_Values = np.array([1, 10, 100, 500]) # Number of Harmonic Oscillators
Therm_Steps = 20000 # Thermalization steps
Block_Size = 10000  # Size of Tau blocks for error analysis

@njit
def WaveFunction(x, alpha): # Definition of Wavefunction
    return np.exp(-0.5 * alpha**2 * (np.sum(x**2, axis=1)))

@njit
def LocalEnergy(x, alpha, D): # Formula for Local Energy E_L(alpha,x)
    return 0.5 * ((m * w**2 - hcut**2 * alpha**4 * m**(-1)) * np.sum(np.sum(x**2, axis=1)) + x.shape[0] * D * alpha**2 * hcut**2 * m**(-1))

def RandomMatrix(N_Cycles, N, D): # Random MC Number generator
    rng = np.random.default_rng(41)
    return np.random.normal(0, 1, (N_Cycles, N, D))

@njit
def DriftForce(x: np.ndarray, alpha: float): # Drift Force formula
    return -alpha**2 * x

@njit
def GF(xOld: np.ndarray, xNew: np.ndarray, F: np.ndarray, alpha): # Formula for the quotient of the Green-Functions
    return np.exp(-(0.5 * (0.25 * Time_Step * (np.sum(DriftForce(xOld, alpha)**2, axis=1) - np.sum(DriftForce(xNew, alpha)**2, axis=1)) + np.sum((xOld - xNew) * (DriftForce(xNew, alpha) - DriftForce(xOld, alpha)), axis=1))))

# MC Algorithm
@njit
def MonteCarloSampling(Alpha, Alpha_Pos, N_Pos, MCMatrix, xOld, MCEnergy, total_steps, rejected_steps, D):
    psiOld = WaveFunction(xOld, Alpha)

    # xOld is a [N x D] matrix
    # Wavefunction is a vector of length N
    # MCEnergy is a matrix [N_Alpha x (N_Cycles-Therm_Steps)]
    
    for j in range(1, N_Cycles): # Start MC Cycle
        xNew = xOld + 0.5 * DriftForce(xOld, Alpha) * Time_Step + MCMatrix[j, :, :] * sqrt(Time_Step)
        psiNew = WaveFunction(xNew, Alpha)

        # Debugging: check that psi is well defined and return flags
        if np.any(psiOld<0):
            return np.zeros(N_Alpha),-1,MCEnergy
        if np.any(psiOld==0):
            return np.zeros(N_Alpha),-2,MCEnergy
            
        # Create a vector of rej/accept steps
        moves = ((psiNew**2 * GF(xOld, xNew, DriftForce(xOld, Alpha), Alpha)) / (psiOld**2)) - np.random.normal(0, 1)
        moves = np.where(moves > 0, 1, 0) # accepted moves become 1, rejected become 0

        # All elements of moves must be 0 for the step to be rejected
        if np.all(moves != 1):
            if j > Therm_Steps: # Update only outside of thermalization
                rejected_steps+=1
                MCEnergy[Alpha_Pos, j - Therm_Steps] = MCEnergy[Alpha_Pos, j - Therm_Steps - 1]
        else:
            if j > Therm_Steps:
                # We add to xOld only the accepted elements of Moves
                matrixmoves = np.ones((xOld.shape[0], D)) * moves[:, np.newaxis] 
                xOld += (xNew - xOld) * matrixmoves
                psiOld = WaveFunction(xOld, Alpha)
                MCEnergy[Alpha_Pos, j - Therm_Steps] = LocalEnergy(xOld, Alpha, D)

    MeanEnergy = np.sum(MCEnergy, axis=1) / (N_Cycles - Therm_Steps)
    return MeanEnergy, rejected_steps, MCEnergy

# Computes the errors
@njit(parallel=True)
def ErrorHandling(MCEnergy_Alpha):
    MeanEnergy = np.sum(MCEnergy_Alpha) / MCEnergy_Alpha.shape[0]
    numerator = 0 # Numerator of C_tau for each alpha
    n = MCEnergy_Alpha.shape[0]
    num_blocks = (n + Block_Size - 1) // Block_Size # Well definiteness checked outside

    for block in prange(num_blocks): 
        start = block * Block_Size # Starting tau 
        end = min((block + 1) * Block_Size, n) # Ending tau

        for tau in range(start, end):
            sum_corr = 0.0
            for i in range(n - tau):
                sum_corr += MCEnergy_Alpha[i] * MCEnergy_Alpha[i + tau] # Not normalized mean <E_i*E_i+tau>

            sum_corr /= (n - tau) # Normalization
            numerator += sum_corr - MeanEnergy**2 # Final expression for tau bar

    return numerator

start_time = time.time() # Start timer
for D in [1, 2, 3]: # Cycle through dimensions
    print(f"Computing Energies in {D} dimensions...")
    
    # Debugging: checking that blocks in error handling are well defined
    if not ((N_Cycles-Therm_Steps)/Block_Size).is_integer():
        print(f"Number of MC Cycles is not a multiple of {Block_Size}: the number of blocks is not an integer")
        sys.exit()
        
    for N_Pos, N in enumerate(N_Values): # Cycle through through N° of Oscillators
        total_steps = 0
        rejected_steps = 0
        alpha_values = np.linspace(0.7, 1.3, N_Alpha) # Range for plotting
        MCEnergy = np.zeros((N_Alpha, N_Cycles - Therm_Steps)) # Initialize Energy matrix
        tau_bar = np.zeros(N_Alpha) # Initializing errors

        for Alpha_Pos in tqdm(range(N_Alpha)): # Cycle through Alphas
            Alpha = alpha_values[Alpha_Pos]
            xOld = Time_Step * (np.random.normal(0, 1, (N, D))) # Set random Non-equilibrium initial position 
            psiOld = WaveFunction(xOld, Alpha)
            MCEnergy[Alpha_Pos, 0] = LocalEnergy(xOld, Alpha, D) # Set Non-equilibrium initial energy 
            MCMatrix = RandomMatrix(N_Cycles, N, D)
            
            #Start MC simulation for each alpha
            MeanEnergy, rejected_steps, MCEnergy = MonteCarloSampling(Alpha, Alpha_Pos, N_Pos, MCMatrix, xOld, MCEnergy, total_steps, rejected_steps, D)
            
            # Debugging for MC: check that psi is well defined
            if rejected_steps==-1:
                raise ValueError("The wavefunction is negative")
                
            if rejected_steps==-2:
                raise ValueError("Division by 0: the Wavefunction is too small")

            # Update errors
            tau_bar[Alpha_Pos] = ErrorHandling(MCEnergy[Alpha_Pos, :])

        # Compute rejection %
        total_steps = (N_Cycles- Therm_Steps) * N_Alpha    
        rejection_percentage = (rejected_steps / total_steps) * 100
        
        # Find Minimum value
        min_index = np.argmin(MeanEnergy)
        best_alpha = alpha_values[min_index]
        min_energy = MeanEnergy[min_index] 
        
        # Set Errors
        err = np.sqrt(np.abs(tau_bar) / (N_Cycles - Therm_Steps))
        min_error = err[min_index] # Error on minimum energy
        
        # Plotting Section
        plt.figure()
        plt.errorbar(alpha_values, MeanEnergy, yerr=err, label=f"N={N}", color='b', marker='o')
        plt.xlabel("Alpha values")
        plt.ylabel("Energy")
        plt.title(f"Energy vs Alpha for N={N}, D={D}")
        plt.legend()
        plt.show()

        # Printing Section
        print(f"Energies in {D} dimensions for {N} harmonic oscillators")
        print(f"Optimal alpha for N={N}: {best_alpha}, Minimum energy: {min_energy}")
        print(f"Error for minimum energy (Alpha = {best_alpha}): {min_error}")
        print(f"Rejection Percentage for N={N}, D={D}: {rejection_percentage:.2f}%")

#Stop Timer
end_time = time.time()
runtime = end_time - start_time
# Print memory and runtime
print(f"Runtime: {runtime:.6f} seconds")
print(f"Memory usage: {get_memory_usage():.2f} MB")
