"""
Two dimensional square lattice.
"""
struct SquareLattice <: AbstractCubicLattice
    L::Int
    sites::Int
    neighs::Matrix{Int} # row = up, right, down, left; col = siteidx
    neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx
    lattice::Matrix{Int}
    bond_lattice::Array{Int, 3}
    bond_info::Matrix{Int}# num bond,src, trg,bond neighs(4)
    # for generic checkerboard decomposition
    # OPT: implement Assaad's two-group square lattice version
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
    bond_checkerboard::Array{Int, 3}
end


# constructors
"""
    SquareLattice(L::Int)

Create a square lattice with linear dimension `L`.
"""
function SquareLattice(L::Int)
    sites = L^2
    lattice = convert(Array, reshape(1:L^2, (L, L)))
    bond_lattice = convert(Array, reshape(1:2*L^2, (L, L,2)))
    neighs, neighs_cartesian = build_neighbortable(SquareLattice, lattice, L)
     bond_info , bond_checkerboard = build_b_neighbortable(SquareLattice, lattice,bond_lattice, L)
    # for generic checkerboard decomposition
    n_bonds = 2*sites
    bonds = zeros(n_bonds, 3)
    bondid = 1
    for src in lattice
        nup = neighs[1, src]
        bonds[bondid,:] .= [src,nup,0]
        bondid += 1

        nright = neighs[2, src]
        bonds[bondid,:] .= [src,nright,0]
        bondid += 1
    end

    return SquareLattice(L, sites, neighs, neighs_cartesian, lattice,
                bond_lattice, bond_info, n_bonds, bonds,bond_checkerboard)
end

function build_neighbortable(::Type{SquareLattice}, lattice, L)
    up = circshift(lattice,(-1,0))
    right = circshift(lattice,(0,-1))
    down = circshift(lattice,(1,0))
    left = circshift(lattice,(0,1))
    neighs = vcat(up[:]',right[:]',down[:]',left[:]')

    neighs_cartesian = Array{Int, 3}(undef, 4, L, L)
    neighs_cartesian[1,:,:] = up
    neighs_cartesian[2,:,:] = right
    neighs_cartesian[3,:,:] = down
    neighs_cartesian[4,:,:] = left
    return neighs, neighs_cartesian
end

function build_b_neighbortable(::Type{SquareLattice}, lattice, bond_lattice, L)
    num_check=div(L^2,2)
    bond_checkerboard=zeros(Int64,3,num_check,4)# 4 typse of Checkerboards
    bond_index = 1:L^2
    right = circshift(lattice,(0,-1))
    b_ul = bond_lattice[:,:,2]
    b_ur = circshift(bond_lattice[:,:,2],(0,-1))
    b_dl = circshift(bond_lattice[:,:,2],(1,0))
    b_dr = circshift(bond_lattice[:,:,2],(1,-1))
    b_n1 = vcat(bond_index[:]',lattice[:]',right[:]',b_ul[:]',b_ur[:]',b_dl[:]',b_dr[:]')
    for i=1:div(L,2)
        bond_checkerboard[:,(i-1)*L+1:i*L,1]=b_n1[1:3,2*(i-1)*L+1:(2*i-1)*L]#horizontal (right) add column
        bond_checkerboard[:,(i-1)*L+1:i*L,2]=b_n1[1:3,(2*i-1)*L+1:2*i*L]#horizontal (right) even column
    end
    bond_index = L^2+1:2*L^2
    up = circshift(lattice,(-1,0))
    b_ul = circshift(bond_lattice[:,:,1],(-1,1))
    b_ur = circshift(bond_lattice[:,:,1],(-1,0))
    b_dl = circshift(bond_lattice[:,:,1],(0,1))
    b_dr = bond_lattice[:,:,1]
    b_n2 = vcat(bond_index[:]',lattice[:]',up[:]',b_ul[:]',b_ur[:]',b_dl[:]',b_dr[:]')
    for i=1:num_check
        bond_checkerboard[:,i,3]=b_n2[1:3,2*i-1] #vertical (up) add
        bond_checkerboard[:,i,4]=b_n2[1:3,2*i] # vertical (up) even
    end
    bond_info=cat(b_n1,b_n2,dims=2)

    return  bond_info, bond_checkerboard
end

@inline nsites(s::SquareLattice) = s.sites
@inline n_bonds(s::SquareLattice) = s.n_bonds
@inline neighbors_lookup_table(s::SquareLattice) = s.neighs
@inline bond_lookup_table(s::SquareLattice) = s.bond_info
@inline bond_checkerboard_table(s::SquareLattice) = s.bond_checkerboard
