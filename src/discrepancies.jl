function kernelized_stein_discrepancy(P, q, K::KernelFunctions.Kernel; ad_backend)
    
    ∇_y(u,v) = ADI.gradient(t->K(u, t), ad_backend, v)
    ∇_x_y(u,v) = ADI.jacobian(t->∇_y(t,v), ad_backend, u)

    function u(x,y)
        k, ∇x_k = ADI.value_and_gradient(t->K(t,y), ad_backend, x)
        ∇y_k = ADI.gradient(t->K(x,t), ad_backend, y)    

        s_x, ∇s_x = LogDensityProblems.logdensity_and_gradient(q, x)
        s_y, ∇s_y = LogDensityProblems.logdensity_and_gradient(q, y)

        ∇s_x'*k*∇s_y + ∇s_x'*∇y_k + ∇x_k'*∇s_y + LinearAlgebra.tr(∇_x_y(x,y))
    end

    D = 0
    n = size(P)[2]

    for i in 1:n
        for j in 1:i
            if i != j
                D += u(P[:, i], P[:, j])
            end
        end
    end

    return D/(n*(n-1))
end
