using LinearAlgebra, Optim, ITensorMPS, HDF5

let
    fn = "/home/andreas/monoaxial_DMI/tdvp/sk1/time_evolved_state.h5"

    @time f = h5open(fn, "r")
    @time psis = [read(f, "psi$i", MPS) for i in 1:1]
end