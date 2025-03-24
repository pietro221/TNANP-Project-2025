import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit, prange
from tqdm import tqdm
import time
import psutil

def get_memory_usage(): # Memory usage function
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)

# Constants
# We use natural units
m = 1
w = 1
hcut = 1

N_Alpha = 40 # Number of Alpha points
N_Cycles = 100000 # Number of Monte Carlo Cycles
Step_Size = np.array([2, 0.6, 0.2, 0.08], dtype=np.float64) # Step size for each N-Harmonic oscillator
N_Values = np.array([1, 10, 100, 500]) # Number of oscillators
Therm_steps = 10000 # Thermalization steps
block_size=10000 # Size of Tau blocks for error analysis

@njit
def WaveFunction(x, alpha): # Definition of Wavefunction
    return np.exp(-alpha**2 * np.sum(np.sum(x**2, axis=1)) / 2)

@njit
def LocalEnergy(x, N, alpha, D): # Formula for local energy E_L(alpha,x)
    return 0.5 * ((m * w**2 - hcut**2 * alpha**4 * m**(-1)) * np.sum(np.sum(x**2, axis=1)) + N * D * alpha**2 * hcut**2 * m**(-1))

def RandomMatrix(N_Cycles, N, D): # Random Monte-Carlo Numbers generator
    return np.random.rand(N_Cycles, N, D)

@njit
def MonteCarloSampling(Alpha, Alpha_Pos, N_Pos, MCMatrix, xOld, MCEnergy, rejected_steps): # MOnte-Carlo Algorithm
    psiOld = WaveFunction(xOld, Alpha)
    
    # xOld is a [N x D] matrix
    # MCEnergy is a [N_Alpha x (N_Cycles-Therm_steps)] matrix
    
    for j in range(1, N_Cycles): # Excluding 0 since the energy at the initial state is computed outside
        if j < Therm_steps: # Start Thermalization
            xNew = xOld + Step_Size[N_Pos] * (MCMatrix[j, :, :] - 0.5) # Updating position
            psiNew = WaveFunction(xNew, Alpha)

            # Debugging for Wavefunction
            if psiOld<0:
                return np.zeros(N_Alpha+1), 0 # Flag for negative wavefunction
            if psiOld==0:
                return np.ones(N_Alpha+1), 0 # Flag for null wavefunction
                
            if (psiNew ** 2 / psiOld ** 2) > np.random.rand(): # Acceptance MC condition
                xOld[:, :] = xNew
                psiOld = psiNew            
        else:
            xNew = xOld + Step_Size[N_Pos] * (MCMatrix[j, :, :] - 0.5)
            psiNew = WaveFunction(xNew, Alpha)
                
            if psiOld<0:
                return np.zeros(N_Alpha+1), 0
            if psiOld**2==0:
                return np.ones(N_Alpha+1), 0 
            
            if (psiNew ** 2 / psiOld ** 2) > np.random.rand(): 
                xOld[:, :] = xNew
                psiOld = psiNew
                MCEnergy[Alpha_Pos, j - Therm_steps] = LocalEnergy(xOld, xOld.shape[0], Alpha, xOld.shape[1]) # Update energy
            else:
                rejected_steps += 1 # Update rejected steps
                MCEnergy[Alpha_Pos, j-Therm_steps] = MCEnergy[Alpha_Pos, j - Therm_steps- 1]

    MeanEnergy = np.sum(MCEnergy, axis=1) / (N_Cycles - Therm_steps) # Compute mean energy vector

    return MeanEnergy, rejected_steps
    
@njit(parallel=True)
def ErrorHandling(MCEnergy_Alpha):
    
    MeanEnergy = np.sum(MCEnergy_Alpha) / (N_Cycles - Therm_steps)
    sigma2 = np.sum((MCEnergy_Alpha - MeanEnergy) ** 2) / (N_Cycles - Therm_steps) # Computes STD**2
    
    numerator = 0 # Numerator of C_tau for each alpha
    n = MCEnergy_Alpha.shape[0]
    
    num_blocks = n // block_size # Well definiteness checked outside
        
    for block in prange(num_blocks):  # Parallelizzazione con prange
        start = block * block_size # Starting tau
        end = min((block + 1) * block_size, n) # Ending tau
        
        for tau in range(start, end):  # Cycle through taus
            sum_corr = 0
            for i in range(n - tau):
                sum_corr += MCEnergy_Alpha[i] * MCEnergy_Alpha[i + tau] # Compute not normalized mean <E_i*E_i+tau>
            
            sum_corr /= (n - tau) # normalization
            numerator += sum_corr - MeanEnergy**2
    
    return numerator # final expression for tau_bar


# --- Main Execution --- #
start_time = time.time() # Start timer

for D in [1,2,3]: # Cycle through dimensions
    
    # Debugging: checking that blocks in error handling are well defined
    if not ((N_Cycles-Therm_steps)/block_size).is_integer():
        print(f"Number of MC Cycles is not a multiple of {block_size}: the number of blocks is not an integer")
        sys.exit()
    
    print(f"Computing Energies in {D} dimensions...")

    for N_Pos, N in enumerate(N_Values): # Cycle through N° of oscillators
        rejected_steps = 0
        alpha_values = np.linspace(0.7, 1.3, N_Alpha) # Range for plotting
        MCEnergy = np.zeros((N_Alpha, N_Cycles-Therm_steps)) # Initialize Energy matrix
        tau_bar=np.zeros(N_Alpha)  # initialize errors
        
        for Alpha_Pos in tqdm(range(N_Alpha)): # Cycle through alphas
            Alpha = alpha_values[Alpha_Pos]
            xOld = Step_Size[N_Pos] * (np.random.rand(N, D) - 0.5) # Set random Non-equilibrium initial position 
            psiOld = WaveFunction(xOld, Alpha)
            MCEnergy[Alpha_Pos, 0] = LocalEnergy(xOld, N, Alpha, D) # Set Non-equilibrium initial energy 
            MCMatrix = RandomMatrix(N_Cycles, N, D)
            
            #Start MC simulation for each alpha
            
            MeanEnergy, rejected_steps = MonteCarloSampling(
                Alpha, Alpha_Pos, N_Pos, MCMatrix, xOld, MCEnergy, rejected_steps,
            )
            
            # Debugging 
            
            if MeanEnergy.shape[0]==N_Alpha+1:
                if MeanEnergy==np.zeros(N_Alpha+1):
                    print("The wavefunction is negative")
                    sys.exit()
                if MeanEnergy==np.ones(N_Alpha+1):
                    print("Division by 0: the Wavefunction is too small")
                    sys.exit()
            if D==3: # Evaluating errors only in 3D case
                tau_bar[Alpha_Pos] = ErrorHandling(MCEnergy[Alpha_Pos,:])

        total_moves = (N_Cycles - Therm_steps) * (N_Alpha) # Total Step of MC
        rejection_percentage = (rejected_steps / total_moves) * 100 
        print(f"Rejection Percentage for N={N}: {rejection_percentage:.2f}%")

        #Find Minimum value
        
        min_index = np.argmin(MeanEnergy)
        best_alpha = alpha_values[min_index]
        min_energy = MeanEnergy[min_index]

        #Set Errors
        err = np.sqrt(np.abs(tau_bar) / (N_Cycles - Therm_steps))
        min_error = err[min_index] # Error on minimum energy
        
        print(f"Optimal alpha for N={N}: {best_alpha}, Minimum energy: {min_energy}")
        
        if D==3:
            print(f"Error for minimum energy (Alpha = {best_alpha}): {min_error}")
        else:
            print("Error not computed for D=/=3")

        # Plotting Section
        plt.figure()
        plt.errorbar(alpha_values, MeanEnergy, yerr=err ,label=f"N={N}", color='b', marker='o')
        plt.xlabel("Alpha values")
        plt.ylabel("Energy")
        plt.title(f"Energy vs Alpha for N={N}, D={D}")
        plt.legend()
        plt.show()
        print(f"Energies in {D} dimensions for {N} harmonic oscillators")

end_time = time.time() #Stop Timer
runtime = end_time - start_time
# Print memory and runtime
print(f"Runtime: {runtime:.6f} seconds")
print(f"Memory usage: {get_memory_usage():.2f} MB") 
