# Bose Einstein Condensation: Monte-Carlo Variational Approach

## Overview

This repository contains three different implementations of a Monte Carlo simulation for a quantum system of harmonic oscillators, focusing on Bose-Einstein condensation. The goal is to study the system's energy behavior by varying parameters such as the number of particles (N), the dimension (D) and the variational parameter (alpha).

The simulations are based on the Variational Monte Carlo (VMC) method, and the results aim to achieve an expected energy of ND/2 with an optimal alpha value of 1.

## Contents

The repository includes three Python scripts, each implementing a different Monte Carlo approach:

- **`1b_errs.py`**: Implements a brute-force Metropolis algorithm for sampling configurations.
- **`1c_errs.py`**: Uses importance sampling to improve efficiency.
- **`1d_errs.py`**: Builds upon the importance sampling framework and incorporates a gradient descent method instead of directly plotting results.

Each script contains both function definitions and executable code, and they can be run independently.

## Requirements

To run the simulations, ensure you have Python installed along with the following dependencies:

```bash
pip install numpy matplotlib

## Usage

It is recommended to run these scripts within a **Jupyter Notebook** for a better experience, as it allows for seamless execution and automatic visualization of the generated plots. Each script contains both function definitions and executable code.

### Steps for running in Jupyter Notebook:

1. Start a Jupyter Notebook session.
2. Create a new cell and copy the contents of one of the scripts (`1b_errs.py`, `1c_errs.py`, or `1d_errs.py`) into the cell.
3. Run the cell to execute the simulation and visualize the resulting plots.

## Expected Results

The simulation should return energy values close to ND/2 when alpha is optimized to approximately 1. If discrepancies are found, it is recommended to check the
implementation of the physical formulas and numerical stability.
