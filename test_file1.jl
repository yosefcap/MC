using LinearAlgebra

N=2000
const A=Diagonal(randn(N))
const x = randn(N,N)
function testmat_A(A,x)

    x*A
end
const c=zeros(N,N)
const B=randn(N)

function testmat_B(c,B)

    for j in 1:N
        c[:,j]=B[j]*x[:,j] ;
    end
end



K=1000

@time for i in 1:K; testmat_A(A,x) ; end
@time for i in 1:K;  testmat_B(c,B) ; end
