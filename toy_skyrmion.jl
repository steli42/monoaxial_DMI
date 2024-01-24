using AddPackage
@add using ITensors
include("spinN.jl")

function main()
    lattice_Q = rectangular(5, 5)
    sites = siteinds("S=N/2", size(lattice_Q, 2), dim=2)
    states = fill("Up", size(lattice_Q, 2))
    ψ = normalize!(productMPS(sites, states)*1.0im)
    @show transpose(hcat(expect(ψ, ["Sx","Sy","Sz"])...))[3,:]
end

main();