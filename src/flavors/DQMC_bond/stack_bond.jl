#type
mutable struct DQMC_bondStack{GreensEltype<:Number, HoppingEltype<:Number} <: AbstractDQMC_bondStack
  eye_flv::Matrix{Float64}
  eye_full::Matrix{Float64}
  ones_vec::Vector{Float64}

  u_stack::Array{GreensEltype, 3}
  d_stack::Matrix{Float64}
  t_stack::Array{GreensEltype, 3}

  Ul::Matrix{GreensEltype}
  Ur::Matrix{GreensEltype}
  Dl::Vector{Float64}
  Dr::Vector{Float64}
  Tl::Matrix{GreensEltype}
  Tr::Matrix{GreensEltype}

  greens::Matrix{GreensEltype}
  greens_temp::Matrix{GreensEltype}
  log_det::Float64 # contains logdet of greens_{mc.p.slices+1} === greens_1
                            # after we calculated a fresh greens in propagate()

  U::Matrix{GreensEltype}
  D::Vector{Float64}
  T::Matrix{GreensEltype}
  udt_tmp::Matrix{GreensEltype}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int # running internally over 0:mc.p.slices+1, where 0 and mc.p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int


  curr_U::Matrix{GreensEltype}
  eV::Matrix{GreensEltype}


  DQMC_bondStack{GreensEltype, HoppingEltype}() where {GreensEltype<:Number, HoppingEltype<:Number} = begin
    @assert isconcretetype(GreensEltype);
    @assert isconcretetype(HoppingEltype);
    new()
  end
end

# type helpers
geltype(::Type{DQMC_bondStack{G,H}}) where {G,H} = G
heltype(::Type{DQMC_bondStack{G,H}}) where {G,H} = H
geltype(mc::DQMC_bond{M, CT, S}) where {M,  CT, S} = geltype(S)
heltype(mc::DQMC_bond{M, CT, S}) where {M,  CT, S} = heltype(S)

# type initialization
function initialize_stack(mc::DQMC_bond)
  GreensEltype = geltype(mc)
  HoppingEltype = heltype(mc)
  N = mc.model.l.sites
  flv = mc.model.flv

  mc.s.eye_flv = Matrix{Float64}(I, flv,flv)
  mc.s.eye_full = Matrix{Float64}(I, flv*N,flv*N)
  mc.s.ones_vec = ones(flv*N)

  mc.s.n_elements = convert(Int, mc.p.slices / mc.p.safe_mult) + 1

  mc.s.u_stack = zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements)
  mc.s.d_stack = zeros(Float64, flv*N, mc.s.n_elements)
  mc.s.t_stack = zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements)

  mc.s.greens = zeros(GreensEltype, flv*N, flv*N)
  mc.s.greens_temp = zeros(GreensEltype, flv*N, flv*N)

  mc.s.Ul = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Ur = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Tl = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Tr = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Dl = ones(Float64, flv*N)
  mc.s.Dr = ones(Float64, flv*N)

  mc.s.U = zeros(GreensEltype, flv*N, flv*N)
  mc.s.D = zeros(Float64, flv*N)
  mc.s.T = zeros(GreensEltype, flv*N, flv*N)
  mc.s.udt_tmp = zeros(GreensEltype, flv*N, flv*N)




  mc.s.ranges = UnitRange[]

  for i in 1:mc.s.n_elements - 1
    push!(mc.s.ranges, 1 + (i - 1) * mc.p.safe_mult:i * mc.p.safe_mult)
  end

  mc.s.curr_U = zero(mc.s.U)
  mc.s.eV = zeros(GreensEltype, flv*N, flv*N)

  # mc.s.hopping_matrix_exp = zeros(HoppingEltype, flv*N, flv*N)
  # mc.s.hopping_matrix_exp_inv = zeros(HoppingEltype, flv*N, flv*N)
  nothing
end
#=
"""
 initializing the blocks of checkerboard hopping matrices.
The matrices are 2×2 blocks of [c s ; s c ]
with c=cosh(Δτ*t(1+ασ)) , s=sinh(Δτ*t(1+ασ))
the indices are [:,:,bond index,slice,checkerboard]
"""
function init_hopping_matrices(mc::DQMC_bond, m::Model)
    conf = mc.conf
    b_ch = m.bond_checkerboard
    dtau = mc.p.delta_tau
    time_slices = mc.p.time_slices
    num_ch = mc.p.num_ch
    num_bond = size(b_ch,2)
    hopping_mat = zeros(2,2,num_bond,time_slices,num_ch)
    t = m.t
    α = m.α

    for i in 1:num_ch
        for j in 1:num_bond
            for k in 1:time_slices
                c = cosh(dtau*t*(1+α*conf[b_ch[1,j,i],k]))
                s = sinh(dtau*t*(1+α*conf[b_ch[1,j,i],k]))
                hopping_mat[:,:,j,k,i] = [c s ; s c]
            end
        end
    end
    return hopping_mat
end
=#
"""
 initializing the blocks of checkerboard hopping matrices.
The matrices are 2×2 blocks of [c s ; s c ]
with c=cosh(Δτ*t(1+ασ)) , s=sinh(Δτ*t(1+ασ))
"""
function init_hopping_matrices(mc::DQMC_bond, m::Model)

    dtau = mc.p.delta_tau
    t = m.t
    α = m.α

    c_u = cosh(dtau*t*(1+α))
    s_u = sinh(dtau*t*(1+α))
    mc.hopping_term_up = [c_u s_u ; s_u c_u]

    c_d = cosh(dtau*t*(1-α))
    s_d = sinh(dtau*t*(1-α))
    mc.hopping_term_down = [c_d  s_d ; s_d  c_d]

end
function init_diag_terms(mc::DQMC_bond, m::Model)
    dtau = mc.p.delta_tau
    μ    = m.μ
    η    = m.η
    L    = m.L
    dims = m.dims
    mc.diag_terms = zeros(L^dims)
    mc.diag_terms_inv = zeros(L^dims)
    for i in 1:L^dims
        disorder = randn()
        mc.diag_terms[i] = exp(-dtau*(μ+η*disorder))
        mc.diag_terms_inv[i] = exp(dtau*(μ+η*disorder))
    end
end

# checkerboard
rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e)<1e-15 ? zero(e) : e,X,X)

# stack construction
"""
Build stack from scratch.
"""
function build_stack(mc::DQMC_bond)
  mc.s.u_stack[:, :, 1] = mc.s.eye_full
  mc.s.d_stack[:, 1] = mc.s.ones_vec
  mc.s.t_stack[:, :, 1] = mc.s.eye_full

  @inbounds for i in 1:length(mc.s.ranges)
        add_slice_sequence_left(mc,i)
    end

  mc.s.current_slice = mc.p.slices + 1
  mc.s.direction = -1

  nothing
end
"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(mc::DQMC_bond, idx::Int)
  copyto!(mc.s.curr_U, mc.s.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", mc.s.ranges[idx])
  for slice in mc.s.ranges[idx]
          multiply_slice_matrix_left!(mc, slice, mc.s.curr_U)
 end

  mc.s.curr_U *= spdiagm(0 => mc.s.d_stack[:, idx])
  mc.s.u_stack[:, :, idx + 1], mc.s.d_stack[:, idx + 1], T = udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx + 1] =  T * mc.s.t_stack[:, :, idx]
end
"""
Updates stack[idx] based on stack[idx+1]

"""
function add_slice_sequence_right(mc::DQMC_bond, idx::Int)
  copyto!(mc.s.curr_U, mc.s.u_stack[:, :, idx + 1])

  for slice in reverse(mc.s.ranges[idx])
          multiply_daggered_slice_matrix_left!(mc, slice, mc.s.curr_U)
  end

  mc.s.curr_U *=  spdiagm(0 => mc.s.d_stack[:, idx + 1])
  mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], T = udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx] = T * mc.s.t_stack[:, :, idx + 1]
end

# Green's function calculation
"""
Calculates G(slice) using mc.s.Ur,mc.s.Dr,mc.s.Tr=B(slice)' ... B(M)' and
mc.s.Ul,mc.s.Dl,mc.s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(mc::DQMC_bond)
    # U, D, T = udt(mc.s.greens) after this
    mc.s.U, mc.s.D, mc.s.T = udt_inv_one_plus(
        UDT(mc.s.Ul, mc.s.Dl, mc.s.Tl),
        UDT(mc.s.Ur, mc.s.Dr, mc.s.Tr),
        tmp = mc.s.U, tmp2 = mc.s.T, tmp3 = mc.s.udt_tmp,
        internaluse = true
    )
    mul!(mc.s.udt_tmp, mc.s.U, Diagonal(mc.s.D))
    mul!(mc.s.greens, mc.s.udt_tmp, mc.s.T)
    mc.s.greens
end

"""
Only reasonable immediately after calculate_greens()!
"""
function calculate_logdet(mc::DQMC_bond)
    mc.s.log_det = real(
        log(complex(det(mc.s.U))) +
        sum(log.(mc.s.D)) +
        log(complex(det(mc.s.T)))
    )
  # mc.s.log_det = real(logdet(mc.s.U) + sum(log.(mc.s.D)) + logdet(mc.s.T))
end

# Green's function propagation
@inline function wrap_greens!(mc::DQMC_bond, gf::Matrix, curr_slice::Int, direction::Int)
  if direction == -1
    multiply_slice_matrix_inv_left!(mc,   curr_slice - 1, gf)
    multiply_slice_matrix_right!(mc,  curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(mc,  curr_slice, gf)
    multiply_slice_matrix_inv_right!(mc, curr_slice, gf)
  end
  nothing
end
function propagate(mc::DQMC_bond)
  if mc.s.direction == 1
    if mod(mc.s.current_slice, mc.p.safe_mult) == 0
      mc.s.current_slice +=1 # slice we are going to
      if mc.s.current_slice == 1
        mc.s.Ur[:, :], mc.s.Dr[:], mc.s.Tr[:, :] = mc.s.u_stack[:, :, 1], mc.s.d_stack[:, 1], mc.s.t_stack[:, :, 1]
        mc.s.u_stack[:, :, 1] = mc.s.eye_full
        mc.s.d_stack[:, 1] = mc.s.ones_vec
        mc.s.t_stack[:, :, 1] = mc.s.eye_full
        mc.s.Ul[:,:], mc.s.Dl[:], mc.s.Tl[:,:] = mc.s.u_stack[:, :, 1], mc.s.d_stack[:, 1], mc.s.t_stack[:, :, 1]

        calculate_greens(mc) # greens_1 ( === greens_{m+1} )
        calculate_logdet(mc)

      elseif 1 < mc.s.current_slice <= mc.p.slices
        idx = Int((mc.s.current_slice - 1)/mc.p.safe_mult)

        mc.s.Ur[:, :], mc.s.Dr[:], mc.s.Tr[:, :] = mc.s.u_stack[:, :, idx+1], mc.s.d_stack[:, idx+1], mc.s.t_stack[:, :, idx+1]
        add_slice_sequence_left(mc, idx)
        mc.s.Ul[:,:], mc.s.Dl[:], mc.s.Tl[:,:] = mc.s.u_stack[:, :, idx+1], mc.s.d_stack[:, idx+1], mc.s.t_stack[:, :, idx+1]

        if mc.p.all_checks
          mc.s.greens_temp = copy(mc.s.greens)
        end

        wrap_greens!(mc, mc.s.greens_temp, mc.s.current_slice - 1, 1)

        calculate_greens(mc) # greens_{slice we are propagating to}

        if mc.p.all_checks
          greensdiff = maximum(abs.(mc.s.greens_temp - mc.s.greens)) # OPT: could probably be optimized through explicit loop
          if greensdiff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.1e\n", mc.s.current_slice, greensdiff)
          end
        end

      else # we are going to mc.p.slices+1
        idx = mc.s.n_elements - 1
        add_slice_sequence_left(mc, idx)
        mc.s.direction = -1
        mc.s.current_slice = mc.p.slices+1 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, mc.s.greens, mc.s.current_slice, 1)
      mc.s.current_slice += 1
    end

  else # mc.s.direction == -1
    if mod(mc.s.current_slice-1, mc.p.safe_mult) == 0
      mc.s.current_slice -= 1 # slice we are going to
      if mc.s.current_slice == mc.p.slices
        mc.s.Ul[:, :], mc.s.Dl[:], mc.s.Tl[:, :] = mc.s.u_stack[:, :, end], mc.s.d_stack[:, end], mc.s.t_stack[:, :, end]
        mc.s.u_stack[:, :, end] = mc.s.eye_full
        mc.s.d_stack[:, end] = mc.s.ones_vec
        mc.s.t_stack[:, :, end] = mc.s.eye_full
        mc.s.Ur[:,:], mc.s.Dr[:], mc.s.Tr[:,:] = mc.s.u_stack[:, :, end], mc.s.d_stack[:, end], mc.s.t_stack[:, :, end]

        calculate_greens(mc) # greens_{mc.p.slices+1} === greens_1
        calculate_logdet(mc) # calculate logdet for potential global update

        # wrap to greens_{mc.p.slices}
        wrap_greens!(mc, mc.s.greens, mc.s.current_slice + 1, -1)

      elseif 0 < mc.s.current_slice < mc.p.slices
        idx = Int(mc.s.current_slice / mc.p.safe_mult) + 1
        mc.s.Ul[:, :], mc.s.Dl[:], mc.s.Tl[:, :] = mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], mc.s.t_stack[:, :, idx]
        add_slice_sequence_right(mc, idx)
        mc.s.Ur[:,:], mc.s.Dr[:], mc.s.Tr[:,:] = mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], mc.s.t_stack[:, :, idx]

        if mc.p.all_checks
          mc.s.greens_temp = copy(mc.s.greens)
        end

        calculate_greens(mc)

        if mc.p.all_checks
          greensdiff = maximum(abs.(mc.s.greens_temp - mc.s.greens)) # OPT: could probably be optimized through explicit loop
          if greensdiff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.1e\n", mc.s.current_slice, greensdiff)
          end
        end

        wrap_greens!(mc, mc.s.greens, mc.s.current_slice + 1, -1)

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(mc, idx)
        mc.s.direction = 1
        mc.s.current_slice = 0 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, mc.s.greens,  mc.s.current_slice, -1)
      mc.s.current_slice -= 1
    end
  end
  nothing
end
