# CheckerboardFalse
const DQMC_CBFalse = DQMC{M, CheckerboardFalse} where M

"""
	slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int, power::Float64=1.)

Direct calculation of effective slice matrix, i.e. no checkerboard.
Calculates `Beff(slice) = exp(−1/2∆tauT)exp(−1/2∆tauT)exp(−∆tauV(slice))`.
"""
function slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int,
					power::Float64=1.)
	eT = mc.s.hopping_matrix_exp
	eTinv = mc.s.hopping_matrix_exp_inv
	eV = mc.s.eV

	interaction_matrix_exp!(mc, m, eV, mc.conf, slice, power)

	if power > 0
		return eT * eT * eV
	else
		return eV * eTinv * eTinv
	end
end


# CheckerboardTrue
#=
const DQMC_CBTrue = DQMC{M, CheckerboardTrue} where M

function slice_matrix(mc::DQMC_CBTrue, m::Model, slice::Int, power::Float64=1.)
  M = eye(heltype(mc), m.flv*m.l.sites)
  if power > 0
    multiply_slice_matrix_left!(mc, m, slice, M)
  else
    multiply_slice_matrix_inv_left!(mc, m, slice, M)
  end
  return M
end
=#
function multiply_slice_matrix_left!(mc::DQMC_bond,  slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
	cb = mod1(slice,4) # !!4 is the number of checkerboards - replace later in a variable!!
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        B = hopping_mat[:,:,n,slice,cb]*[M[i,:]' ; M[j,:]']
	        M[i,:] = B[1,:]
	        M[j,:] = B[2,:]
	    end
	end
    nothing
end
function multiply_slice_matrix_right!(mc::DQMC_bond,  slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
	cb = mod1(slice,4) # !!4 is the number of checkerboards - replace later in a variable!!
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        B = [M[i,:]  M[j,:]]*hopping_mat[:,:,n,slice,cb]
	        M[:,i] = B[:,1]
	        M[:,j] = B[:,2]
	    end
	end
    nothing
end

function multiply_slice_matrix_inv_left!(mc::DQMC_bond,  slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
	cb = mod1(slice,4) # !!4 is the number of checkerboards - replace later in a variable!!
    h_inv = zeros(Float64,2,2)
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]

	        h_inv[1,1]=hopping_mat[1,1,n,slice,cb]
	        h_inv[2,2]=hopping_mat[2,2,n,slice,cb]
	        h_inv[1,2]=-hopping_mat[1,2,n,slice,cb]
	        h_inv[2,1]=-hopping_mat[2,1,n,slice,cb]
	        B = h_inv*[M[i,:]' ; M[j,:]']
	        M[i,:] = B[1,:]
	        M[j,:] = B[2,:]
	    end
	end
    nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_bond, slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
	cb = mod1(slice,4) # !!4 is the number of checkerboards - replace later in a variable!!
    h_inv = zeros(Float64,2,2)
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        h_inv[1,1]=hopping_mat[1,1,n,slice,cb]
	        h_inv[2,2]=hopping_mat[2,2,n,slice,cb]
	        h_inv[1,2]=-hopping_mat[1,2,n,slice,cb]
	        h_inv[2,1]=-hopping_mat[2,1,n,slice,cb]
	        B = [M[i,:]  M[j,:]]*h_inv
	        M[:,i] = B[:,1]
	        M[:,j] = B[:,2]
	    end
	end
    nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_bond, slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
	cb = mod1(slice,4) # !!4 is the number of checkerboards - replace later in a variable!!
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        B = adjoint(hopping_mat[:,:,n,slice,cb])*[M[i,:]' ; M[j,:]']
	        M[i,:] = B[1,:]
	        M[j,:] = B[2,:]
	    end
	end
    nothing
end
function multiply_daggered_slice_matrix_right!(mc::DQMC_bond,  slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
	cb = mod1(slice,4) # !!4 is the number of checkerboards - replace later in a variable!!
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        B = [M[i,:]  M[j,:]]*adjoint(hopping_mat[:,:,n,slice,cb])
	        M[:,i] = B[:,1]
	        M[:,j] = B[:,2]
	    end
	end
    nothing
end



#=
function multiply_slice_matrix_inv_left!(mc::DQMC_bond, cb::Int, slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
    h_inv = zeros(Float64,2,2)
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]

	        h_inv[1,1]=hopping_mat[1,1,n,slice,cb]
	        h_inv[2,2]=hopping_mat[2,2,n,slice,cb]
	        h_inv[1,2]=-hopping_mat[1,2,n,slice,cb]
	        h_inv[2,1]=-hopping_mat[2,1,n,slice,cb]
	        B = h_inv*[M[i,:]' ; M[j,:]']
	        M[i,:] = B[1,:]
	        M[j,:] = B[2,:]
	    end
	end
    nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_bond, cb::Int, slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
    h_inv = zeros(Float64,2,2)
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        h_inv[1,1]=hopping_mat[1,1,n,slice,cb]
	        h_inv[2,2]=hopping_mat[2,2,n,slice,cb]
	        h_inv[1,2]=-hopping_mat[1,2,n,slice,cb]
	        h_inv[2,1]=-hopping_mat[2,1,n,slice,cb]
	        B = [M[i,:]  M[j,:]]*h_inv
	        M[:,i] = B[:,1]
	        M[:,j] = B[:,2]
	    end
	end
    nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_bond, cb::Int, slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        B = adjoint(hopping_mat[:,:,n,slice,cb])*[M[i,:]' ; M[j,:]']
	        M[i,:] = B[1,:]
	        M[j,:] = B[2,:]
	    end
	end
    nothing
end
function multiply_daggered_slice_matrix_right!(mc::DQMC_bond,  cb::Int, slice::Int, M::AbstractMatrix{T}) where T<:Number
    bond_ch = mc.model.bond_checkerboard
    hopping_mat = mc.hopping_mat#[:,:,n,slice,cb]
    N=size(hopping_mat,3) #number of bonds in checkerboard
    @inbounds @views begin
	    for n in 1:N
	        i = bond_ch[2,n,cb]
	        j = bond_ch[3,n,cb]
	        B = [M[i,:]  M[j,:]]*adjoint(hopping_mat[:,:,n,slice,cb])
	        M[:,i] = B[:,1]
	        M[:,j] = B[:,2]
	    end
	end
    nothing
end
=#
