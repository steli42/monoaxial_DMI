using DataFrames, CSV
# from spherical to cartesian coordinates
function s2c(r, t, p)
    return r .* [sin(t) * cos(p), sin(t) * sin(p), cos(t)]
end

# from cartesian to spherical coordinates
function c2s(spin)
    r = norm(spin)
    t = acos(spin[3] / r)
    p = sign(spin[2]) * acos(spin[1] / norm(spin[1:2]))
    return [r, t, p]
end

function lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    df = DataFrame()
    xs = lattice[1, :]
    ys = lattice[2, :]
    zs = lattice[3, :]
    Sxs = real.(spins[1, :])
    Sys = real.(spins[2, :])
    Szs = real.(spins[3, :])
    if p["boundary_conditions"] == "classical_environment"
        for (idal, al) in enumerate(aux_lattices)
            for i in axes(al, 2)
                for (x, y, z) in zip(lattice[1, :], lattice[2, :], lattice[3, :])
                    dist = [x, y, z] .- al[:, i]
                    if abs(norm(dist)) < 1.2
                        # check if the element already exists
                        b1 = xs .== al[1, i]
                        b2 = ys .== al[2, i]
                        if any(b1 .* b2)
                            continue
                        end
                        push!(xs, al[1, i])
                        push!(ys, al[2, i])
                        push!(zs, al[3, i])
                        push!(Sxs, real.(0.5 * 𝐦[1, idal, i]))
                        push!(Sys, real.(0.5 * 𝐦[2, idal, i]))
                        push!(Szs, real.(0.5 * 𝐦[3, idal, i]))
                    end
                end
            end
        end
    end

    df[!, "x"] = xs
    df[!, "y"] = ys
    df[!, "z"] = zs
    df[!, "S_x"] = Sxs
    df[!, "S_y"] = Sys
    df[!, "S_z"] = Szs
    return df
end

function lobs_arr_to_df(lattice, aux_lattices, spins_arr, 𝐦, p; T=1.0, lbl="n")
    df_all = DataFrame()
    dt = T/length(spins_arr)
    for (ids, spins) in enumerate(spins_arr)
        df = DataFrame()
        xs = lattice[1, :]
        ys = lattice[2, :]
        zs = lattice[3, :]
        Sxs = real.(spins[1, :])
        Sys = real.(spins[2, :])
        Szs = real.(spins[3, :])
        if p["boundary_conditions"] == "classical_environment"
            for (idal, al) in enumerate(aux_lattices)
                for i in axes(al, 2)
                    for (x, y, z) in zip(lattice[1, :], lattice[2, :], lattice[3, :])
                        dist = [x, y, z] .- al[:, i]
                        if abs(norm(dist)) < 1.2
                            # check if the element already exists
                            b1 = xs .== al[1, i]
                            b2 = ys .== al[2, i]
                            if any(b1 .* b2)
                                continue
                            end
                            push!(xs, al[1, i])
                            push!(ys, al[2, i])
                            push!(zs, al[3, i])
                            push!(Sxs, real.(0.5 * 𝐦[1, idal, i]))
                            push!(Sys, real.(0.5 * 𝐦[2, idal, i]))
                            push!(Szs, real.(0.5 * 𝐦[3, idal, i]))
                        end
                    end
                end
            end
        end
        df[!, lbl] = ones(size(xs))*ids*dt
        df[!, "x"] = xs
        df[!, "y"] = ys
        df[!, "z"] = zs
        df[!, "S_x"] = Sxs
        df[!, "S_y"] = Sys
        df[!, "S_z"] = Szs

        df_all = vcat(df_all, df; cols = :union)
    end
    # @show df_all
    return df_all
end

function corr_to_df(lattice, corr, p)
    df = DataFrame()
    x1s = []
    y1s = []
    z1s = []
    x2s = []
    y2s = []
    z2s = []
    for i1 in axes(lattice,2), i2 in axes(lattice,2)
        push!(x1s, lattice[1,i1])
        push!(x2s, lattice[1,i2])
        push!(y1s, lattice[2,i1])
        push!(y2s, lattice[2,i2])
        push!(z1s, lattice[3,i1])
        push!(z2s, lattice[3,i2])
    end
    df[!, "x"] = x1s
    df[!, "y"] = y1s
    df[!, "z"] = z1s
    df[!, "x'"] = x2s
    df[!, "y'"] = y2s
    df[!, "z'"] = z2s
    for k in keys(corr)
        df[!, "$(k[1])*$(k[2])_re"] = real.(vcat(corr[k]...))
        df[!, "$(k[1])*$(k[2])_im"] = imag.(vcat(corr[k]...))
    end
    return df
end