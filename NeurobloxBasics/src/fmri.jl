
"""
    BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0)

Create a balloon model blox which computes the hemodynamic responses to some underlying neuronal activity
The formal definition of this blox is:
```math
    \\frac{ds}{dt} = \\text{jcn} - \\kappa s - \\gamma (u-1)
    \\frac{du}{dt} = s
    \\frac{d\\nu}{dt} = u - v^{1/\\alpha}
    \\frac{dq}{dt} = u E(u, E_0)/E_0 - v^{1/\\alpha} q/v
```
where ``jcn`` is any input to the blox (represents the neuronal activity).

Includes also the bold measurement function:
```math
    \\lambda(\\nu, q) = V_0 \\left[ k_1 (1-q) + k_2 \\left( 1 - \\frac{q}{v} \\right) + k_3 (1-v)\\right]
```
    
Arguments:
- `name`: Name given to `ODESystem` object within the blox.
- `namespace`: Additional namespace above `name` if needed for inheritance.
- `lnκ`: logarithmic prefactor to signal decay H[1], set to 0 for standard parameter value.
- `lnτ`: logarithmic prefactor to transit time H[3], set to 0 for standard parameter value.
- `lnϵ`: logarithm of ratio of intra- to extra-vascular signal

NB: the prefix ln of the variables u, ν, q as well as the parameters κ, τ denotes their transformation into logarithmic space
to enforce their positivity. This transformation is considered in the derivates of the model equations below. 

Citations:
1. Stephan K E, Weiskopf N, Drysdale P M, Robinson P A, and Friston K J. Comparing Hemodynamic Models with DCM. NeuroImage 38, no. 3 (2007): 387–401. doi: 10.1016/j.neuroimage.2007.07.040
2. Hofmann D, Chesebro A G, Rackauckas C, Mujica-Parodi L R, Friston K J, Edelman A, and Strey H H. Leveraging Julia's Automated Differentiation and Symbolic Computation to Increase Spectral DCM Flexibility and Speed, 2023. doi: 10.1101/2023.10.27.564407
"""
@blox struct BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0, lnϵ=0.0) <: AbstractObserver
    #= hemodynamic parameters
        H[1] - signal decay                                   d(ds/dt)/ds)
        H[2] - autoregulation                                 d(ds/dt)/df)
        H[3] - transit time                                   (t0)
        H[4] - exponent for Fout(v)                           (alpha)
        H[5] - resting state oxygen extraction                (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    #=
        NB: Biophysical constants for 1.5T scanners
        
        B[1]: TE, time to echo
        B[2]: V0, resting venous volume (%)
        B[3]: r0, slope of intravascular relaxation rate R_iv as a function of oxygen saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
        B[4]: nu0, frequency offset at the outer surface of magnetized vessels (Hz)
        B[5]: E0, resting oxygen extraction fraction
    =#
    B = [0.04, 4, 25, 40.3, 0.4]
    k1 = 4.3*B[1]*B[4]*B[5]                         # coefficients in BOLD signal model

    @params lnκ lnτ lnϵ H B k1                      # assign the modified parameters
    @states s=0.0 lnu=0.0 lnν=0.0 lnq=0.0
    @inputs jcn=0.0
    @equations begin
        D(s)   = jcn - H[1]*exp(lnκ)*s - H[2]*(exp(lnu) - 1)
        D(lnu) = s / exp(lnu)
        D(lnν) = (exp(lnu) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(lnν))
        D(lnq) = (exp(lnu)/exp(lnq)*((1 - (1 - H[5])^(exp(lnu)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ))
    end
    @computed_properties begin
        bold   = B[2]*(k1 - k1*exp(lnq) + exp(lnϵ)*B[3]*B[5]*B[1] - exp(lnϵ)*B[3]*B[5]*B[1]*exp(lnq)/exp(lnν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν))
    end
end

# """
#     boldsignal(;name, lnϵ=0.0)

#     Bold signal observer function. This requires connection to the variables ν and q of a balloon model.
#     The formal definition of this blox is:
#     ```math
#         \\lambda(\\nu, q) = V_0 \\left[ k_1 (1-q) + k_2 \\left( 1 - \\frac{q}{v} \\right) + k_3 (1-v)\\right]
#     ```

# Arguments:
# - `name`: Name given to `ODESystem` object within the blox.
# - lnϵ  : logarithm of ratio of intra- to extra-vascular signal

# NB: the prefix ln of the variables ν, q as well as the parameters ϵ denotes their transformation into logarithmic space
# to enforce their positivity.

# Citations:
# 1. Stephan K E, Weiskopf N, Drysdale P M, Robinson P A, and Friston K J. Comparing Hemodynamic Models with DCM. NeuroImage 38, no. 3 (2007): 387–401. doi: 10.1016/j.neuroimage.2007.07.040
# 2. Hofmann D, Chesebro A G, Rackauckas C, Mujica-Parodi L R, Friston K J, Edelman A, and Strey H H. Leveraging Julia's Automated Differentiation and Symbolic Computation to Increase Spectral DCM Flexibility and Speed, 2023. doi: 10.1101/2023.10.27.564407

# """
# function boldsignal(;name, lnϵ=0.0)
#     # NB: Biophysical constants for 1.5T scanners
#     # Time to echo
#     TE  = 0.04
#     # resting venous volume (%)
#     V0  = 4
#     # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
#     # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
#     r0  = 25
#     # frequency offset at the outer surface of magnetized vessels (Hz)
#     nu0 = 40.3
#     # resting oxygen extraction fraction
#     E0  = 0.4
#     # Coefficients in BOLD signal model
#     k1  = 4.3*nu0*E0*TE

#     params = @parameters lnϵ=lnϵ
#     vars = @variables bold(t) lnν(t) lnq(t)

#     eqs = [
#         bold ~ V0*(k1 - k1*exp(lnq) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq)/exp(lnν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν))
#     ]

#     ODESystem(eqs, t, vars, params; name=name)
# end

# """
# Simulated BOLD response to input

#     This is the version of the Balloon-Windkessel model used in Endo et al. (2020), and subsequently in Antal et al. (2023) and van Niuewenhuizen et al. (2023).
#     As it is only a convolution after the simulation of the neural activity, it is not a block for inclusion in the ODESystem.

# """
# function boldsignal_endo_balloon(t, y, TR=800, dt_orig=1)

#     # Parameters are from Endo et al. (2020).
#     κ = 0.65;
#     α = 0.32;
#     τ = 0.98;
#     ρ = 0.34;
#     V_0 = 0.02;
#     γ = 0.41;

#     bold = zeros(size(y));
    
#     for roi in 1:length(y[:, 1])
#         s = ones(1, length(y[1, :])).*0.1;
#         f = ones(size(s)).*0.1;
#         v = ones(size(s)).*0.1;
#         q = ones(size(s)).*0.1;

#         for i in 2:length(y[1, :])
#             z = y[roi, i];
#             dt = t[i]-t[i-1];
#             ds = z - (κ*s[i-1])-(γ*(f[i-1]-1));
#             df = s[i-1];
#             dv = (f[i-1]-(v[i-1]^(1/α)))/τ;
#             dq = (((f[i-1]*(1-(1-(ρ^(1/f[i-1])))))/ρ)-(((v[i-1]^(1/α))*(q[i-1]))/v[i-1]))/τ;
#             s[i] = s[i-1] + ds*dt;
#             f[i] = f[i-1] + df*dt;
#             v[i] = v[i-1] + dv*dt;
#             q[i] = q[i-1] + dq*dt;
#         end
#         bold[roi, :] = V_0 .* ((7 .* ρ .* (1 .- q ./ v)) .+ ((2 .* ρ .- 0.2) .* (1 .-v )));
#     end
    
#     return (bold[:, 1:Int(round(TR/dt_orig)):end])
# end
