fn = "/Users/andreas/gits/monoaxial_DMI/tdvp/sk/state.h5"

f = h5open(fn, "r")
psi = read(f, "psi", MPS)
close(f)

inner(conj.(psi), psi)