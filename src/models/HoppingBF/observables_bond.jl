"""
    prepare_observables(m::HoppingBFModel)

Initializes observables for the Hopping BF model and returns a `Dict{String, Observable}`.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
@inline function prepare_observables(mc::DQMC_bond, m::HoppingBFModel)
    obs = Dict{String,Observable}()
    obs["confs"] = Observable(HBFConf, "Configurations")
    obs["greens"] = Observable(typeof(mc.s.greens), "Equal-times Green's function")
    obs["Eboson"] = Observable(Float64, "Bosonic energy")

    return obs
end

"""
    measure_observables!(m::HoppingBFModel, obs::Dict{String,Observable}, conf::HBFConf, E::Float64)

Measures observables and updates corresponding `Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
@noinline function measure_observables!(mc::DQMC_bond, m::HoppingBFModel,
							obs::Dict{String,Observable}, conf::HBFConf)
    push!(obs["confs"], mc.conf)
    push!(obs["greens"], mc.s.greens)
    push!(obs["Eboson"], energy_boson(mc, m, conf))
    nothing
end

"""
    measure_observables!(mc::DQMC_bond, m::HoppingBFModel, obs::Dict{String,Observable})

Finish measurements of observables.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
@inline function finish_observables!(mc::DQMC_bond, m::HoppingBFModel,
							obs::Dict{String,Observable})
    nothing
end
