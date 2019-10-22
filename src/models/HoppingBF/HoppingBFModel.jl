const HBFConf = Array{Int8, 2} # conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HBFDistribution = Int8[-1,1]

"""
    HoppingBFModel(; dims, L[, kwargs...])

Create an Hopping Boson Fermion model on `dims`-dimensional cubic lattice
with linear system size `L`. Additional allowed `kwargs` are:

 * `mu::Float64=0.0`: chemical potential
 * `α::Float64 = 1.0`: Boson-Fermion coupling
 * `t::Float64=1.0`: hopping energy
"""

@with_kw_noshow struct HoppingBFModel{C<:AbstractCubicLattice} <: Model
    # user mandatory
    dims::Int
    L::Int
    J::Float64 #ising coupling
    h::Float64 #transvers field

    # user optional
    t::Float64 = 1.0
    α::Float64 = 0.0
    μ::Float64 = 0.0
    η::Float64 = 0.0 #disorder

    # non-user fields
    l::C = choose_lattice(HoppingBFModel, dims, L)
    neighs::Matrix{Int} = neighbors_lookup_table(l)
    bond_info::Matrix{Int} = bond_lookup_table(l)
    bond_checkerboard::Array{Int, 3} = bond_checkerboard_table(l)
    flv::Int = 1
end

function choose_lattice(::Type{HoppingBFModel}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    HoppingBFModel(params::Dict)
    HoppingBFModel(params::NamedTuple)

Create an Hopping Boson-Fermion Model with (keyword) parameters as specified in the
dictionary/named tuple `params`.
"""
HoppingBFModel(params::Dict{Symbol, T}) where T = HoppingBFModel(; params...)
HoppingBFModel(params::NamedTuple) = HoppingBFModel(; params...)

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HoppingBFModel) = "$(model.dims)D Hopping Boson-Fermion Model"
Base.show(io::IO, model::HoppingBFModel) = print(io, "$(model.dims)D Hopping Boson-Fermion Model, L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, m::MIME"text/plain", model::HoppingBFModel) = print(io, model)





# implement `Model` interface
@inline nsites(m::HoppingBFModel) = nsites(m.l)
@inline n_bonds(m::HoppingBFModel) = n_bonds(m.l)

# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC_bond}, m::HoppingBFModel, nslices::Int) = rand(HBFDistribution, n_bonds(m), nslices)



@inline function propose_local(mc::DQMC_bond, m::HoppingBFModel, n::Int, time_slice::Int, conf::HBFConf)
    # see for example dos Santos (2002)
    greens = mc.s.greens
    dtau = mc.p.delta_tau
    Kxy = mc.K_xy
    Ktau = mc.K_tau
    α = m.α
    num_slices = mc.p.time_slices
    bond_info = m.bond_info
    forwards = mod(time_slice,num_slices)+1
    backwards = mod(time_slice-2,num_slices)+1

    c_α = cosh(2*α)-1
    s_α = sinh(2*α)
    if conf[n,time_slice]==1 #+ to -
        Δ  = [c_α  -s_α ; -s_α  c_α] #flipping + spin to - spin
        ΔE_boson = -2*(Kxy*(conf[bond_info[4,n],time_slice]+conf[bond_info[5,n],time_slice]+
        conf[bond_info[6,n],time_slice]+conf[bond_info[7,n],time_slice])
        +Ktau*(conf[n,forwards]+conf[n,backwards]))
    else
        Δ = [c_α   s_α ;  s_α  c_α]#flipping - spin to + spin
        ΔE_boson = 2*(Kxy*(conf[bond_info[4,n],time_slice]+conf[bond_info[5,n],time_slice]+
        conf[bond_info[6,n],time_slice]+conf[bond_info[7,n],time_slice])
        +Ktau*(conf[n,forwards]+conf[n,backwards]))
    end
    i = m.bond_info[2,n]
    j = m.bond_info[3,n]
    g = [1-greens[i,i] -greens[i,j] ; -greens[j,i] 1-greens[j,j]]

    detratio = det(I + Δ*g)^2 # squared because of two spin sectors.
    return detratio, ΔE_boson, Δ,g
end

@inline function accept_local!(mc::DQMC_bond, m::HoppingBFModel, n::Int, cb::Int,
            slice::Int, conf::HBFConf, Δ, detratio, ΔE_boson::Float64)
    greens = mc.s.greens
    t = m.t
    α = m.α
    i = m.bond_info[2,n]
    j = m.bond_info[3,n]
    g = [1-greens[i,i] -greens[i,j] ; -greens[j,i] 1-greens[j,j]]
    Λred = inv( inv(Δ) + g )
    Λ = zeros(size(green))
    Λ[i,i] = Λred[1,1]
    Λ[i,j] = Λred[1,2]
    Λ[j,i] = Λred[2,1]
    Λ[j,j] = Λred[2,2]
    greens = greens*(I-Λ*greens)

    conf[n, slice] *= -1
    c = cosh(dtau*t*(1+α*conf[n, slice]))
    s = sinh(dtau*t*(1+α*conf[n, slice]))
    mc.hopping_mat[:,:,n,slice,cb] = [c s ; s c]

    nothing
end





# implement DQMC interface: optional
"""
Green's function is real for the attractive Hubbard model.
"""
@inline greenseltype(::Type{DQMC_bond}, m::HoppingBFModel) = Float64


"""
Calculate energy contribution of the boson, i.e. Hubbard-Stratonovich/Hirsch field.
"""
#=
@inline function energy_boson(m::HubbardModelAttractive, hsfield::HubbardConf)
  dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end
=#
include("observables_bond.jl")
