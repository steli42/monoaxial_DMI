
"""
space(::SiteType"S=N/2", N = float)

Create the Hilbert space for a site of type "S=N/2".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(::SiteType"S=N/2"; dim=2)
    return dim
end

function generate_spin_matrices(;dim=2)

    # initialize spin matrices
    Sx = zeros(ComplexF64,dim,dim);
    Sy = zeros(ComplexF64,dim,dim);
    Sz = zeros(ComplexF64,dim,dim);

    spin = 0.5*(dim-1)

    # construct Sx, Sy and Sz
    for idx = 1 : dim

        for idy = 1 : dim

            entryXY = 0.5 * sqrt((spin + 1) * (idx + idy - 1) - idx * idy);

            if (idx + 1) == idy
                Sx[idx,idy] += entryXY;
                Sy[idx,idy] -= 1im * entryXY;
            end

            if idx == (idy + 1)
                Sx[idx,idy] += entryXY;
                Sy[idx,idy] += 1im * entryXY;
            end

            if idx == idy
                Sz[idx,idy] += spin + 1 - idx;
            end

        end

    end

    # compute Id, Sp and Sm
    Id = one(Sz);
    Sp = Sx + 1im * Sy;
    Sm = Sx - 1im * Sy;

    spin_matrices = Dict("Sx" => Sx, "Sy" => Sy, "Sz" => Sz, "S+" => Sp, "S-" => Sm, "Id" => Id)

    # return spin matrices
    return spin_matrices

end

_op(::OpName"Id", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Id"]
_op(::OpName"Sx", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sx"]
_op(::OpName"Sy", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sy"]
_op(::OpName"Sz", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"]
_op(::OpName"S+", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["S+"]
_op(::OpName"S-", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["S-"]

function ITensors.op(on::OpName, st::SiteType"S=N/2", s::ITensors.Index)
    return itensor(_op(on, st; dim=dim(s)), s', dag(s))
end

ITensors.val(::ValName"Up", ::SiteType"S=N/2"; dim=2) = 1
ITensors.val(::ValName"Dn", ::SiteType"S=N/2"; dim=2) = dim
ITensors.val(::ValName"↑", ::SiteType"S=N/2"; dim=2) = 1
ITensors.val(::ValName"↓", ::SiteType"S=N/2"; dim=2) = dim
ITensors.val(::ValName"0", ::SiteType"S=N/2"; dim=2) = 1
ITensors.val(::ValName"1", ::SiteType"S=N/2"; dim=2) = dim
ITensors.state(::StateName"Up", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"][:, 1]
ITensors.state(::StateName"Dn", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"][:, end]
ITensors.state(::StateName"↑", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"][:, 1]
ITensors.state(::StateName"↓", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"][:, end]
ITensors.state(::StateName"0", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"][:, 1]
ITensors.state(::StateName"1", ::SiteType"S=N/2"; dim=2) = generate_spin_matrices(;dim=dim)["Sz"][:, end]