"""
fmri.jl 

Models of fMRI signals.

BalloonModel : computes hemodynamic responses
boldsignal   : computes BOLD signal
"""

"""
    BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0)

    Create a balloon model blox which computes the hemodynamic responses to some underlying neuronal activity
    The formal definition of this blox is:
 """
    # ```math
    #     \\frac{ds}{dt} = \\text{jcn} - \\kappa s - \\gamma (u-1)
    #     \\frac{du}{dt} = s
    #     \\frac{d\\nu}{dt} = u - v^{1/\\alpha}
    #     \\frac{dq}{dt} = u E(u, E_0)/E_0 - v^{1/\\alpha} q/v
    # ```
"""
    where ``jcn`` is any input to the blox (represents the neuronal activity)

Arguments:
- `name`: Name given to `ODESystem` object within the blox.
- `namespace`: Additional namespace above `name` if needed for inheritance.
- `lnκ`: logarithmic prefactor to signal decay H[1], set to 0 for standard parameter value.
- `lnτ`: logarithmic prefactor to transit time H[3], set to 0 for standard parameter value.

NB: the prefix ln of the variables u, ν, q as well as the parameters κ, τ denotes their transformation into logarithmic space
to enforce their positivity. This transformation is considered in the derivates of the model equations below. 

Citations:
1. Stephan K E, Weiskopf N, Drysdale P M, Robinson P A, and Friston K J. Comparing Hemodynamic Models with DCM. NeuroImage 38, no. 3 (2007): 387–401. doi: 10.1016/j.neuroimage.2007.07.040
2. Hofmann D, Chesebro A G, Rackauckas C, Mujica-Parodi L R, Friston K J, Edelman A, and Strey H H. Leveraging Julia's Automated Differentiation and Symbolic Computation to Increase Spectral DCM Flexibility and Speed, 2023. doi: 10.1101/2023.10.27.564407

"""
struct BalloonModel <: ObserverBlox
    params
    output
    jcn
    odesystem
    namespace
    function BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0)
        #= hemodynamic parameters
            H(1) - signal decay                                   d(ds/dt)/ds)
            H(2) - autoregulation                                 d(ds/dt)/df)
            H(3) - transit time                                   (t0)
            H(4) - exponent for Fout(v)                           (alpha)
            H(5) - resting state oxygen extraction                (E0)
        =#

        H = [0.64, 0.32, 2.00, 0.32, 0.4]
        # p = progress_scope(lnκ, lnτ)                  # progress scope if needed
        p = paramscoping(lnκ=lnκ, lnτ=lnτ)  # finally compile all parameters
        lnκ, lnτ = p                                  # assign the modified parameters
        
        sts = @variables s(t)=1.0 lnu(t)=1.0 lnν(t)=1.0 [output=true, description="hemodynamic_observer"] lnq(t)=1.0 [output=true, description="hemodynamic_observer"] jcn(t)=0.0 [input=true]

        eqs = [
            D(s)   ~ jcn - H[1]*exp(lnκ)*s - H[2]*(exp(lnu) - 1),
            D(lnu) ~ s / exp(lnu),
            D(lnν) ~ (exp(lnu) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(lnν)),
            D(lnq) ~ (exp(lnu)/exp(lnq)*((1 - (1 - H[5])^(exp(lnu)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ))
        ]
        sys = System(eqs, t, name=name)
        new(p, Num(0), sts[5], sys, namespace)
    end
end


"""
    boldsignal(;name, lnϵ=0.0)

    Bold signal observer function. This requires connection to the variables ν and q of a balloon model.
    The formal definition of this blox is:
 """
    # ```math
    #     \\lambda(\\nu, q) = V_0 \\left[ k_1 (1-q) + k_2 \\left( 1 - \\frac{q}{v} \\right) + k_3 (1-v)\\right]
    # ```
"""

Arguments:
- `name`: Name given to `ODESystem` object within the blox.
- lnϵ  : logarithm of ratio of intra- to extra-vascular signal

NB: the prefix ln of the variables ν, q as well as the parameters ϵ denotes their transformation into logarithmic space
to enforce their positivity.

Citations:
1. Stephan K E, Weiskopf N, Drysdale P M, Robinson P A, and Friston K J. Comparing Hemodynamic Models with DCM. NeuroImage 38, no. 3 (2007): 387–401. doi: 10.1016/j.neuroimage.2007.07.040
2. Hofmann D, Chesebro A G, Rackauckas C, Mujica-Parodi L R, Friston K J, Edelman A, and Strey H H. Leveraging Julia's Automated Differentiation and Symbolic Computation to Increase Spectral DCM Flexibility and Speed, 2023. doi: 10.1101/2023.10.27.564407

"""
function boldsignal(;name, lnϵ=0.0)
    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE

    params = @parameters lnϵ=lnϵ
    vars = @variables bold(t) lnν(t) lnq(t)   # TODO: got to be really careful with the current implementation! A simple permutation of this breaks the algorithm!

    eqs = [
        bold ~ V0*(k1 - k1*exp(lnq) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq)/exp(lnν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν))
    ]

    ODESystem(eqs, t, vars, params; name=name)
end
