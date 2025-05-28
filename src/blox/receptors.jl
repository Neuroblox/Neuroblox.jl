
struct MoradiFullNMDAR <: AbstractReceptor
    params
    system
    namespace

    function MoradiFullNMDAR(;name,
                    namespace=nothing,
                    E=-0.7, # mV
                    k=0.007, # mV⁻¹
                    V_0=-100, # mV
                    τ_A=1.69,
                    τ_B0=3.97,
                    τ_C0=41.62,
                    τ_g=1,
                    w_B=0.65 ,
                    λ_B=0.0243,
                    λ_C=0.01,
                    a_B=0.7,
                    a_C=34.69,
                    Mg_O=1, # mM
                    IC_50=4.1, # mM
                    T=37, # Celsius
                    F=96485.332, # Faraday constant, C/mol
                    R=8.314, # Gas constant, J/(K*mol)
                    z=2, # Magnesium valence
                    δ=0.8,
                    Q_A_10=2.2,
                    T_A_0=31.5, # Celsius
                    Q_B_10=3.68,
                    T_B_0=35, # Celsius
                    Q_C_10=2.65,
                    T_C_0=35, # Celsius
                    Q_g_10=1.52,
                    T_g_0=26, # Celsius
                    spk_coeff=1
    )
        w_C = 1 - w_B

        p = paramscoping(E=E, k=k, V_0=V_0, τ_A=τ_A, τ_B0=τ_B0, τ_C0=τ_C0, τ_g=τ_g, w_B=w_B, w_C=w_C, λ_B=λ_B, λ_C=λ_C, a_B=a_B, a_C=a_C, Mg_O=Mg_O, IC_50=IC_50, T=T, F=F, R=R, z=z, δ=δ, Q_A_10=Q_A_10, T_A_0=T_A_0, Q_B_10=Q_B_10, T_B_0=T_B_0, Q_C_10=Q_C_10, T_C_0=T_C_0, Q_g_10=Q_g_10, T_g_0=T_g_0, spk_coeff=spk_coeff)
        E, k, V_0, τ_A, τ_B0, τ_C0, τ_g, w_B, w_C, λ_B, λ_C, a_B, a_C, Mg_O, IC_50, T, F, R, z, δ, Q_A_10, T_A_0, Q_B_10, T_B_0, Q_C_10, T_C_0, Q_g_10, T_g_0, spk_coeff = p

        sts = @variables begin 
            A(t)=0
            B(t)=0
            C(t)=0
            g(t)=0
            V(t)
            [input=true]
            jcn(t)
            [input=true]
            I(t)
        end

        time_constant_B(V) = τ_B0 + a_B * (1 - exp(λ_B * V))
        time_constant_C(V) = τ_C0 + a_C * (1 - exp(λ_C * V))
        temp_modifier(T, Q_10, T_0) = Q_10^((T_0 - T) / 10)  
        g_final(V) = k * (V - V_0)

        eqs = [
                D(A) ~ spk_coeff*jcn - A / (temp_modifier(T, Q_A_10, T_A_0) * τ_A),
                D(B) ~ spk_coeff*jcn - B / (temp_modifier(T, Q_B_10, T_B_0) * time_constant_B(V)),
                D(C) ~ -C / (temp_modifier(T, Q_C_10, T_C_0) * time_constant_C(V)),
                D(g) ~ (w_C * C + w_B * B)*(g_final(V) - g) / (temp_modifier(T, Q_g_10, T_g_0) * τ_g),
                I ~ (w_C * C + w_B * B - A) * g * (V - E) * (1 / (1 + Mg_O * exp(z * δ * F * V / (R * T)) / IC_50))
        ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
    end
end

struct MoradiNMDAR <: AbstractReceptor
    params
    system
    namespace   

    function MoradiNMDAR(;name,
                    namespace=nothing,
                    E=-0.7, # mV
                    k=0.007, # mV⁻¹
                    V_0=-100, # mV
                    g_VI=1,
                    τ_A=1.47,
                    τ_B=391.64,
                    τ_g=50,
                    Mg_O=1, # mM
                    IC_50=4.1, # mM
                    T=295.15, # K
                    F=96485.332, # Faraday constant, C/mol
                    R=8.314, # Gas constant, J/(K*mol)
                    z=2, # Magnesium valence
                    δ=0.8,
                    spk_coeff=1
    )
        p = paramscoping(E=E, k=k, V_0=V_0, g_VI=g_VI, τ_A=τ_A, τ_B=τ_B, τ_g=τ_g, Mg_O=Mg_O, IC_50=IC_50, T=T, F=F, R=R, z=z, δ=δ, spk_coeff=spk_coeff)
        E, k, V_0, g_VI, τ_A, τ_B, τ_g, Mg_O, IC_50, T, F, R, z, δ, spk_coeff = p

        sts = @variables begin 
            A(t)=0
            B(t)=0
            g(t)=0
            V(t)
            [input=true]
            jcn(t)
            [input=true]
            I(t)=0 # remove this initial guess so MTK does not throw an unbalanced system warning
        end

        g_VD(V) = k * (V - V_0)

        eqs = [
                D(A) ~ spk_coeff*jcn - A / τ_A,
                D(B) ~ spk_coeff*jcn - B / τ_B,
                D(g) ~ B * (g_VD(V) - g) / τ_g,
                I ~ (B - A) * (g_VI + g) * (V - E) * (1 / (1 + Mg_O * exp(- z * δ * F * V / (R * T)) / IC_50))
        ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
    end
end
