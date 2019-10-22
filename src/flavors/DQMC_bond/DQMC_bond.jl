include("abstract_bond.jl")

"""
Analysis data of determinant quantum Monte Carlo (DQMC) simulation
"""
@with_kw mutable struct DQMC_bondAnalysis
    acc_rate::Float64 = 0.
    prop_local::Int = 0
    acc_local::Int = 0
    acc_rate_global::Float64 = 0.
    prop_global::Int = 0
    acc_global::Int = 0
end

"""
Parameters of determinant quantum Monte Carlo (DQMC)
"""
@with_kw struct DQMC_bondParameters
    global_moves::Bool = false
    global_rate::Int = 5
    thermalization::Int = 100 # number of thermalization sweeps
    num_ch::Int = 4
    sweeps::Int = num_ch*25 # number of sweeps (after thermalization)
    all_checks::Bool = true # e.g. check if propagation is stable/instable
    safe_mult::Int = num_ch*2

    delta_tau::Float64 = 0.1
    beta::Float64

    time_slices::Int = beta / delta_tau
    slices::Int  = num_ch*time_slices
    @assert isinteger(beta / delta_tau) string("beta/delta_tau", "
        (= number of imaginary time slices) must be an integer but is",
        beta / delta_tau, ".")

    measure_every_nth::Int = 10
end

"""
Determinant quantum Monte Carlo (DQMC_bond) simulation
"""
#mutable struct DQMC_bond{M<:Model, CB<:Checkerboard, ConfType<:Any,
mutable struct DQMC_bond{M<:Model, ConfType<:Any,
            Stack<:AbstractDQMC_bondStack} <: MonteCarloFlavor
    model::M
    conf::ConfType
    hopping_mat::Array{Float64,5}
    diag_terms::Array{Float64,1} #diagonal terms including chemical potential and onsite disorder
    diag_terms_inv::Array{Float64,1} #inverse diagonal terms including chemical potential and onsite disorder
    s::Stack

    p::DQMC_bondParameters
    a::DQMC_bondAnalysis
    obs::Dict{String, Observable}

    K_xy::Float64
    K_tau::Float64
    DQMC_bond{M, ConfType, Stack}() where {M<:Model, ConfType<:Any, Stack<:AbstractDQMC_bondStack} = new()
end

include("stack_bond.jl")
include("slice_matrices_bond.jl")

"""
    DQMC_bond(m::M; kwargs...) where M<:Model

Create a determinant quantum Monte Carlo simulation for model `m` with
keyword parameters `kwargs`.
"""
function DQMC_bond(m::M; seed::Int=-1, kwargs...) where M<:Model

    p = DQMC_bondParameters(; kwargs...)
    geltype = greenseltype(DQMC_bond, m)
    conf = rand(DQMC_bond, m, p.slices)
    mc = DQMC_bond{M, typeof(conf), DQMC_bondStack{geltype,Float64}}()
    mc.model = m
    mc.p = p
    mc.s = DQMC_bondStack{geltype,Float64}()
    mc.K_xy = m.J*p.delta_tau
    mc.K_tau = 0.5*log(coth(m.h*p.delta_tau))
    init!(mc, seed=seed, conf=conf)
    return mc
end


"""
    DQMC(m::M, params::Dict)
    DQMC(m::M, params::NamedTuple)

Create a determinant quantum Monte Carlo simulation for model `m` with
(keyword) parameters as specified in the dictionary/named tuple `params`.
"""
DQMC_bond(m::Model, params::Dict{Symbol, T}) where T = DQMC_bond(m; params...)
DQMC_bond(m::Model, params::NamedTuple) = DQMC_bond(m; params...)


# convenience
@inline beta(mc::DQMC_bond) = mc.p.beta
@inline nslices(mc::DQMC_bond) = mc.p.slices
@inline model(mc::DQMC_bond) = mc.model
@inline conf(mc::DQMC_bond) = mc.conf
@inline current_slice(mc::DQMC_bond) = mc.s.current_slice


# cosmetics
import Base.summary
import Base.show
Base.summary(mc::DQMC_bond) = "DQMC_bond simulation of $(summary(mc.model))"
function Base.show(io::IO, mc::DQMC_bond)
    print(io, "Determinant quantum Monte Carlo simulation\n")
    print(io, "Model: ", mc.model, "\n")
    print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, sigdigits=3)))")
end
Base.show(io::IO, m::MIME"text/plain", mc::DQMC_bond) = print(io, mc)


"""
    init!(mc::DQMC[; seed::Real=-1])

Initialize the determinant quantum Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `Random.seed!(seed)`.
"""
function init!(mc::DQMC_bond; seed::Real=-1, conf=rand(DQMC_bond,model(mc),nslices(mc)))
    seed == -1 || Random.seed!(seed)

    mc.conf = conf
    mc.hopping_mat = init_hopping_matrices(mc, mc.model)
    init_diag_terms(mc, mc.model)
    initialize_stack(mc)
    mc.obs = prepare_observables(mc, mc.model)
    mc.a = DQMC_bondAnalysis()
    nothing
end

"""
    run!(mc::DQMC[; verbose::Bool=true, sweeps::Int, thermalization::Int])

Runs the given Monte Carlo simulation `mc`.
Progress will be printed to `stdout` if `verbose=true` (default).
"""
function run!(mc::DQMC_bond; verbose::Bool=true, sweeps::Int=mc.p.sweeps,
    thermalization=mc.p.thermalization)
    total_sweeps = sweeps + thermalization

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    # fresh stack
    verbose && println("Preparing Green's function stack")
    initialize_stack(mc) # redundant ?!
    build_stack(mc)

    propagate(mc)


    _time = time()
    verbose && println("\n\nThermalization stage - ", thermalization)
    for i in 1:total_sweeps
        verbose && (i == thermalization + 1) &&
            println("\n\nMeasurement stage - ", sweeps)
        for u in 1:2 * nslices(mc)
            update(mc, i)

            if i > thermalization && current_slice(mc) == nslices(mc) &&
                    mc.s.direction == -1 && (i-1)%mc.p.measure_every_nth == 0
                measure_observables!(mc, mc.model, mc.obs, mc.conf)
            end
        end

        if mod(i, 10) == 0
            mc.a.acc_rate = mc.a.acc_rate / (10 * 2 * nslices(mc))
            mc.a.acc_rate_global = mc.a.acc_rate_global / (10 / mc.p.global_rate)
            sweep_dur = (time() - _time)/10
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", sweep_dur)
                @printf("\t\tacc rate (local) : %.1f%%\n", mc.a.acc_rate*100)
                if mc.p.global_moves
                  @printf("\t\tacc rate (global): %.1f%%\n", mc.a.acc_rate_global*100)
                  @printf("\t\tacc rate (global, overall): %.1f%%\n",
                    mc.a.acc_global/mc.a.prop_global*100)
                end
            end

            mc.a.acc_rate = 0.0
            mc.a.acc_rate_global = 0.0
            flush(stdout)
            _time = time()
        end
    end
    finish_observables!(mc, mc.model, mc.obs)

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)

    nothing
end

"""
    update(mc::DQMC, i::Int)

Propagates the Green's function and performs local and global updates at
current imaginary time slice.
"""
function update(mc::DQMC_bond, i::Int)
    propagate(mc)
    sweep_spatial(mc)

    nothing
end

"""
    sweep_spatial(mc::DQMC_bond,cb::Int)

Performs a sweep of local moves along spatial dimension at current
imaginary time slice.
"""
function sweep_spatial(mc::DQMC_bond)

    m = model(mc)
    N = div(nsites(m),2) # number of bonds in a checkerboard
    bond_checkerboard = m.bond_checkerboard
    cs = current_slice(mc)
    num_ch=mc.p.num_ch
    time_slice = cld(cs,num_ch) # imagenary time
    cb = mod1(cs,num_ch) # number of checkerboard in time slice
    @inbounds for b in 1:N
        i = bond_checkerboard[1,b,cb] # index of bond i in checkerboard ch
        detratio, ΔE_boson, Δ = propose_local(mc, m, i, time_slice, conf(mc))# y.c. added index j
        mc.a.prop_local += 1
        if abs(imag(detratio)) > 1e-6
            println("Did you expect a sign problem? imag. detratio: ",
                abs(imag(detratio)))
            @printf "%.10e" abs(imag(detratio))
        end
        p = real(exp(- ΔE_boson) * detratio)

        # Metropolis
        if p > 1 || rand() < p
            accept_local!(mc, m, i, cb, time_slice, conf(mc), Δ, detratio,
                ΔE_boson)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
        end
    end
    nothing
end
#=
"""
    greens(mc::DQMC)

Obtai the current equal-time Green's function.

Internally, `mc.s.greens` is an effective Green's function. This method transforms
this effective one to the actual Green's function by multiplying hopping matrix
exponentials from left and right.
"""
function greens(mc::DQMC_CBFalse)
    eThalfminus = mc.s.hopping_matrix_exp
    eThalfplus = mc.s.hopping_matrix_exp_inv

    greens = copy(mc.s.greens)
    greens .= greens * eThalfminus
    greens .= eThalfplus * greens
    return greens
end
function greens(mc::DQMC_CBTrue)
    chkr_hop_half_minus = mc.s.chkr_hop_half
    chkr_hop_half_plus = mc.s.chkr_hop_half_inv

    greens = copy(mc.s.greens)

    @inbounds @views begin
        for i in reverse(1:mc.s.n_groups)
          greens .= greens * chkr_hop_half_minus[i]
        end
        for i in reverse(1:mc.s.n_groups)
          greens .= chkr_hop_half_plus[i] * greens
        end
    end
    return greens
end
=#
include("DQMC_bond_mandatory.jl")
include("DQMC_bond_optional.jl")
