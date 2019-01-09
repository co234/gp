using LinearAlgebra

# Squared exponential kernel (rbf)
function se_kernel(xi::Array{Float64}, xj::Array{Float64},s_f,l)
    m = length(xi)
    n = length(xj)
    k = zeros(m,n)
    for i in range(1,m)
        for j in range(1,n)
            k[i,j] = exp(-(xi[i]-xj[j])^2/(2*l^2))*s_f^2
        end
    end
    return k
end

# periodical kernel
function per_kernel(xi,xj,s_f,l,b3,b4)
    m = length(xi)
    n = length(xj)
    k = zeros(m,n)
    for i in range(1,m)
        for j in range(1,n)
            k[i,j] = exp(-2*(sin(pi*(xi[i]-xj[j])/b3))^2/l^2 + (xi[i]-xj[j])^2/(b4^2))*s_f^2
        end
    end
    return k
end

# radial quadratic kernel
function RQ_kernel(xi,xj,s_f,l,b3)
    m = length(xi)
    n = length(xj)
    k = zeros(m,n)
    for i in range(1,m)
        for j in range(1,n)
            k[i,j] = s_f^2*(1+(xi[i] - xj[j])^2/(2*l*(b3^2)))^(-b3)
        end
    end
    return k
end

# combined rbf, periodical and radial quadratic kernels
function combined_kernel(xi,xj,theta)

    if length(theta[1]) == 1
        s_f1 = theta[1]
        s_f2 = s_f1
        s_f3 = s_f1
    else
        s_f1,s_f2,s_f3 = theta[1][1],theta[1][2],theta[1][3]
    end

    l = theta[2]
    s_n = theta[3]
    b3 = theta[4]
    b4 = theta[5]

    return se_kernel(xi,xj,s_f1,l)+per_kernel(xi,xj,s_f2,l,b3,b4)+RQ_kernel(xi,xj,s_f3,l,b3)
end


function gp_inference(x,y,x_new,theta)
    s_n = theta[3]

    k_x,k_xt,k_xtxt = combined_kernel(x,x,theta), combined_kernel(x,x_new,theta),combined_kernel(x_new,x_new,theta)

    k_inv = pinv(k_x+(s_n^2)*Matrix{Float64}(I,size(k_x,1),size(k_x,1)))

    xt_mean = (k_xt')*k_inv*y
    xt_cov = k_xtxt - k_xt'*k_inv*k_xt

    return(xt_mean,xt_cov)
end
