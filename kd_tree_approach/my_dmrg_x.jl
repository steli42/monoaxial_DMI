using ITensors: array, contract, dag, uniqueind, onehot
using ITensorMPS: MPS
using LinearAlgebra: eigen
using KrylovKit: eigsolve

function eigen_updater(operator, state; internal_kwargs)
    contracted_operator = contract(operator, ITensor(true))

    d, u = eigen(contracted_operator; ishermitian=true)

    u_ind = uniqueind(u, contracted_operator)
    u′_ind = uniqueind(d, u)
    max_overlap, max_index = findmax(abs, array(state * dag(u)))
    u_max = u * dag(onehot(eltype(u), u_ind => max_index))
    d_max = d[u′_ind=>max_index, u_ind=>max_index]
    return u_max, (; eigval=d_max)
end

function my_eigen_updater(operator, state; internal_kwargs)
    # contracted_operator = contract(operator, ITensor(true))

    dim = min(size(operator)[1], 100)
    # dim = size(operator)[1]

    d, u = eigsolve(operator, state, dim, :SR, krylovdim=300)
    # @show d

    max_overlap, max_index = findmax(norm, [state * dag(t) for t in u])
    u_max = u[max_index]
    d_max = d[max_index]
    return u_max, (; eigval=d_max)
end

function my_dmrg_x(
    operator, state::MPS; updater=eigen_updater, (observer!)=ITensorMPS.default_observer(), kwargs...
)
    info_ref = Ref{Any}()
    info_observer = ITensorMPS.values_observer(; info=info_ref)
    observer = ITensorMPS.compose_observers(observer!, info_observer)
    eigvec = ITensorMPS.alternating_update(operator, state; updater, (observer!)=observer, kwargs...)
    return info_ref[].eigval, eigvec
end
