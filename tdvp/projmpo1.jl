import ITensorMPS.AbstractProjMPO
import ITensors.@debug_check
import ITensors.@timeit_debug
import ITensors.@printf
"""
A ProjMPO1 computes and stores the projection of an
MPO into a basis defined by an MPS, leaving a
certain number of site indices of the MPO unprojected.
Which sites are unprojected can be shifted by calling
the `position!` method.

Drawing of the network represented by a ProjMPO1 `P(H)`,
showing the case of `nsite(P)==2` and `position!(P,psi,4)`
for an MPS `psi`:

```
o--o--o--o-   -o--o--o--o--o--o <psi|
|  |  |  |  |  |  |  |  |  |  |
o--o--o--o--o--o--o--o--o--o--o H
|  |  |  |  |  |  |  |  |  |  |
o--o--o--o-   -o--o--o--o--o--o |psi>
```
"""
mutable struct ProjMPO1 <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
end
ProjMPO1(H::MPO) = ProjMPO1(0, length(H) + 1, 1, H, Vector{ITensor}(undef, length(H)))
