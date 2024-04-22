using ITensors, Printf, PyPlot, ITensors.HDF5
pygui(true)
include("functions.jl")

let 
    L = 15 

    nsweeps = 100
    maxdim = [100 for n=1:nsweeps]
    cutoff = 1E-10
    loadPsi = true #true loads a chosen .h5 file (the name of the file needs to be specified in functions.jl -- Change that later its really annoying )

    obs = DMRGObserver(; energy_tol = 1e-10, minsweeps = 10)

    J = -1.0 
    D = 2*pi/L
    Bcr = 0.5*D*D
    Bpin = 2.0
    α = 0.14

    sites, ψ₀ = get_Initial_Psi(L,"skyrmion",loadPsi,"0_14_Mag2D_original.h5") 

    H = build_Hamiltonian(sites, D, Bpin, Bcr, J, α, L)
        
    E₁, ψ₁ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = obs)

    σ₁ = inner(H, ψ₁, H, ψ₁) - E₁^2 
    
    Magx01 = expect(ψ₁,"Sx")
    Magy01 = expect(ψ₁,"Sy")
    Magz01 = expect(ψ₁,"Sz")
        
    f_original = open("refined_state.csv", "w")
    for (j,mz) in enumerate(Magz01)
    @printf f_original "%f,"  (j-1.0) ÷ L
    @printf f_original "%f,"  (j-1.0) % L
    @printf f_original "%f,"  0.0
    @printf f_original "%f,"  Magx01[j]
    @printf f_original "%f,"  Magy01[j]
    @printf f_original "%f,"  Magz01[j]
    @printf f_original "%f\n" sqrt(Magx01[j]^2 + Magy01[j]^2 + Magz01[j]^2)
    end  
    close(f_original)

    println("For alpha = $α: Final energy of psi = $E₁")
    println("For alpha = $α: Final energy variance of psi = $σ₁")
    @show(norm(inner(ψ₀,ψ₁)))

    # here we calculate the excited state

    E_exc, ψ_exc = dmrg(H,[ψ₁],ψ₀; nsweeps, maxdim, cutoff) #might be better to try out different initial states here

    σ_exc = inner(H, ψ_exc, H, ψ_exc) - E_exc^2 
    
    Magx_exc = expect(ψ_exc,"Sx")
    Magy_exc = expect(ψ_exc,"Sy")
    Magz_exc = expect(ψ_exc,"Sz")
        
    f_original = open("excited_state.csv", "w")
    for (j,mz) in enumerate(Magz_exc)
    @printf f_original "%f,"  (j-1.0) ÷ L
    @printf f_original "%f,"  (j-1.0) % L
    @printf f_original "%f,"  0.0
    @printf f_original "%f,"  Magx_exc[j]
    @printf f_original "%f,"  Magy_exc[j]
    @printf f_original "%f,"  Magz_exc[j]
    @printf f_original "%f\n" sqrt(Magx_exc[j]^2 + Magy_exc[j]^2 + Magz_exc[j]^2)
    end  
    close(f_original)

    println("For alpha = $α: Final energy of excited psi = $E_exc")
    println("For alpha = $α: Final energy variance of psi = $σ_exc")
    @show(norm(inner(ψ_exc,ψ₁)))
    
end