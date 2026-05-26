# Example Simulation for "Skyrmionic Schrödinger cat states in monoaxial chiral magnets"

We recommend to use julia version 1.12 and the following versions of julia packages:
- ITensors v0.9.13
- ITensorMPS v0.3.22
- NDTensors v0.4.15
- HDF5 v1.14.6
- Observers v0.2.5
- JSON v1.1.0
- NearestNeighbors v0.4.22
- ProgressBars v1.5.1
- Adapt v4.4.0
- KrylovKit v0.10.2
- TupleTools v1.6.0
- DataFrames v1.8.0
- CSV v0.10.15

The main script is given by `minimal_example.jl`.
Configurations for the initial states, DMRG-X and TDVP algorithms are contained in `cfg/default.json`.
The main script can be called within the Julia REPL, or by executing `julia minimal_example.jl` in the command line.

The simulation will then initialize a single skyrmion in the center of a $19\times15$ lattice embedded in a polarized environment.

After performing the DMRG-x algorithm until convergence, the gradient Zeeman field is applied and TDVP is performed to simulate the time-dependent wave function.

If the script finishes successfully, the output folder contains the following files associated with the results of the DMRG-x simulation: two-point correlations in `corr.csv`, information on the final energy in `energy.csv` and local magnetization profiles in `lobs.csv`.
Associated with the TDVP simulations, the energy drift during the evolution is presented in `series_energy.csv`, and the time series of the magnetization profile in `series_lobs.csv`.
