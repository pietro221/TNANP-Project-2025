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
# We use Natural Units
m = 1
w = 1
hcut = 1

N_Cycles = 1000 # N° of MC cycles
Time_Step = 0.01 # Same time step for every N
N_Values = np.array([1, 10, 100, 500]) # N°of Harmonic Oscillators
Therm_Steps = 200 # N° of Thermalization steps
Delta = 1e-10 # Tolerance
Lambda = 0.0001 # Learning rate of GDM
Block_Size = 100 # Size of Tau blocks for error analysis

@njit
def WaveFunction(x, alpha): # Definition of Wave function
    return np.exp(-0.5 * alpha**2 * (np.sum(x**2, axis=1)))

@njit
def LocalEnergy(x, alpha, D): # Definition of local energy
    return 0.5 * ((m * w**2 - hcut**2 * alpha**4 * m**(-1)) * np.sum(np.sum(x**2, axis=1)) + x.shape[0] * D * alpha**2 * hcut**2 * m**(-1))

def RandomMatrix(N_Cycles, N, D): # random number generator
    rng = np.random.default_rng(41)
    return np.random.normal(0, 1, (N_Cycles, N, D))

@njit
def DriftForce(x: np.ndarray, alpha: float): # Formula for drift force
    return -alpha**2 * x

@njit
def GF(xOld: np.ndarray, xNew: np.ndarray, F: np.ndarray, alpha): # Quotient of Green-functions
    return np.exp(-(0.5 * (0.25 * Time_Step * (np.sum(DriftForce(xOld, alpha)**2, axis=1) - np.sum(DriftForce(xNew, alpha)**2, axis=1)) + np.sum((xOld - xNew) * (DriftForce(xNew, alpha) - DriftForce(xOld, alpha)), axis=1))))

@njit
def MonteCarloSampling(Alpha, MCMatrix, xOld, rejected_steps, D): # MC algorithm
    
    MCEnergy = np.zeros(N_Cycles - Therm_Steps) # Generate vector of energies
    psiOld = WaveFunction(xOld, Alpha)
    MCEnergy[0]=LocalEnergy(xOld, Alpha, D) # Set starting energy
    
    for j in range(1, N_Cycles): # Start MC Cycle
        xNew = xOld + 0.5 * DriftForce(xOld, Alpha) * Time_Step + MCMatrix[j, :, :] * sqrt(Time_Step)
        psiNew = WaveFunction(xNew, Alpha)

        # Debugging: check that psi is well defined
        if np.any(psiOld<0):
            return 0,-1,MCEnergy # Return flags
        if np.any(psiOld==0):
            return 0,-2,MCEnergy

        # Create a vector of rej/accept steps
        moves = (((psiNew**2) * GF(xOld, xNew, DriftForce(xOld, Alpha), Alpha)) / ((psiOld**2))) - np.random.normal(0, 1)
        moves = np.where(moves > 0, 1, 0)
        
        # All elements of moves must be 0 for the step to be rejected 
        if np.all(moves != 1):
            if j > Therm_Steps: # Update only outside of thermalization
                rejected_steps += 1
                MCEnergy[j - Therm_Steps] = MCEnergy[j - Therm_Steps - 1]
        else:
            if j > Therm_Steps:
                # We add to xOld only the accepted elements of Moves
                matrixmoves = np.ones((N, D)) * moves[:, np.newaxis]
                xOld += (xNew - xOld) * matrixmoves
                psiOld = WaveFunction(xOld + (xNew - xOld) * matrixmoves, Alpha)
                MCEnergy[j - Therm_Steps] = LocalEnergy(xOld, Alpha, D)

    MeanEnergy = np.sum(MCEnergy) / (N_Cycles - Therm_Steps)
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
    
#--- Main Execution ---#
start_time = time.time() # Start timer
for D in [1, 2, 3]: # Cycle through dimensions
    
    # Debugging: checking that blocks in error handling are well defined
    if not ((N_Cycles-Therm_Steps)/block_size).is_integer():
        print(f"Number of MC Cycles is not a multiple of {block_size}: the number of blocks is not an integer")
        sys.exit()
    
    print(f"Computing Energies in {D} dimensions...")
    
    for N in N_Values: # Cycle through Oscillators
        print(f"Starting gradient descent for N = {N}")
        
        # Set numbers of iterations for each oscillator number
        max_iterations = 400 if N == 1 else 200 if N == 10 else 50 if N == 100 else 40 if N == 500 else 0
        optimal_alphas = []

        for initial_alpha in [0.95, 1.05]: # Setting initial guesses for the descent
            rejected_steps = 0
            alpha = initial_alpha
            MCMatrix = RandomMatrix(N_Cycles, N, D)
            xOld = Time_Step * (np.random.normal(0, 1, (N, D))) # Set random non-equilibrium initial position 

            # Cycle through iterations
            for iteration in tqdm(range(max_iterations), desc=f"N={N}, alpha_initial={initial_alpha}", unit="iteration"):
                
                energy_up, rejected_steps,_ = MonteCarloSampling(alpha + 0.01, MCMatrix, xOld, rejected_steps, D)
                
                # Debugging for MC: check that psi is well defined
                if rejected_steps==-1:
                    raise ValueError("The wavefunction is negative")
                if rejected_steps==-2:
                    raise ValueError("Division by 0: the Wavefunction is too small")
                
                energy_down, rejected_steps,_= MonteCarloSampling(alpha - 0.01, MCMatrix, xOld, rejected_steps, D)
                
                if rejected_steps==-1:
                    raise ValueError("The wavefunction is negative")
                if rejected_steps==-2:
                    raise ValueError("Division by 0: the Wavefunction is too small")
                    
                gradient = (energy_up - energy_down) / 0.02 #Calculata gradient
                
                if abs(gradient) < Delta: # Check if the gradient is small enough
                    print(f"Converged after {iteration + 1} iterations.")
                    break
                alpha = alpha - Lambda * gradient
                # Ensure Alpha stays in the range
                if alpha < 0.7:
                    alpha = 0.7
                if alpha > 1.3:
                    alpha = 1.3

            optimal_alphas.append(alpha)

        #Calculate ground state energy once Optimal alpha is set
        optimal_alpha = np.mean(optimal_alphas)
        MCMatrix = RandomMatrix(N_Cycles, N, D)
        xOld = Time_Step * (np.random.normal(0, 1, (N, D)))
        ground_state_energy, rejected_steps_ground,MCEnergy = MonteCarloSampling(optimal_alpha, MCMatrix, xOld, 0, D)
        
        # Debugging for MC: check that psi is well defined
        if rejected_steps==-1:
            raise ValueError("The wavefunction is negative")
        if rejected_steps==-2:
            raise ValueError("Division by 0: the Wavefunction is too small")

        # Computing Rejection rate for ground state
        total_steps= N_Cycles-Therm_Steps
        rejection_rate = (rejected_steps_ground / total_steps) * 100

        # Computing Errors
        tau_bar = ErrorHandling(MCEnergy)
        err = sqrt(abs(tau_bar) / (N_Cycles - Therm_Steps))
        
        # Printing section
        data_to_print = [optimal_alpha, ground_state_energy, rejection_rate, err]
        headers = [f"OPTIMAL ALPHA FOR N={N}, D={D}", f"GROUND STATE ENERGY FOR N={N}, D={D}", f"REJECTION RATE FOR GROUND STATE ENERGY AT N={N}, D={D} in %", f"ERROR FOR N={N}, D={D}]"]
        print(tabulate([data_to_print], headers=headers, tablefmt="grid"))

end_time = time.time() # Stop counting Running Time
runtime = end_time - start_time
print(f"Total runtime: {runtime:.6f} seconds")

print(f"Memory usage: {get_memory_usage():2f} MB") # print memory usage
