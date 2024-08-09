abstract type AbstractPINGNeuron <: AbstractNeuronBlox end

struct PINGNeuronExci <: AbstractPINGNeuron
    params
    output
    jcn
    odesystem
    namespace
    function PINGNeuronExci(;name,
                             namespace=nothing,
                             C=1.0,
                             g_Na=100.0,
                             V_Na=50.0,
                             g_K=80.0,
                             V_K=-100.0,
                             g_L=0.1,
                             V_L=-67.0,
                             I_ext=0.0)
        p = paramscoping(C=C, g_Na=g_Na, V_Na=V_Na, g_K=g_K, V_K=V_K, g_L=g_L, V_L=V_L, I_ext=I_ext)
        C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext = p
        sts = @variables V(t)=0.0 [output=true] n(t)=0.0 h(t)=0.0 jcn(t)=0.0 [input=true]
        
        a_m(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
        b_m(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
        a_n(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
        b_n(v) = 0.5*exp((v+57.0)/40.0)
        a_h(v) = 0.128*exp((v+50.0)/18.0)
        b_h(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
        
        m∞(v) = a_m(v)/(a_m(v) + b_m(v))
        eqs = [D(V) ~ g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext,
               D(n) ~ (a_n(V)*(1.0 - n) - b_n(V)*n),
               D(h) ~ (a_h(V)*(1.0 - h) - b_h(V)*h)
        ]
        sys = ODESystem(eqs, t, sts, p; name=name)
        new(p, sts[1], sts[4], sys, namespace)
    end
end

struct PINGNeuronInhib <: AbstractPINGNeuron
    params
    output
    jcn
    odesystem
    namespace
    function PINGNeuronInhib(;name,
                             namespace=nothing,
                             C=1.0,
                             g_Na=35.0,
                             V_Na=55.0,
                             g_K=9.0,
                             V_K=-90.0,
                             g_L=0.1,
                             V_L=-65.0,
                             I_ext=0.0)
        p = paramscoping(C=C, g_Na=g_Na, V_Na=V_Na, g_K=g_K, V_K=V_K, g_L=g_L, V_L=V_L, I_ext=I_ext)
        C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext = p
        sts = @variables V(t)=0.0 [output=true] n(t)=0.0 h(t)=0.0 jcn(t)=0.0 [input=true]

        a_m(v) = 0.1*(v+35.0)/(1.0 - exp(-(v+35.0)/10.0))
        b_m(v) = 4*exp(-(v+60.0)/18.0)
        a_n(v) = 0.05*(v+34.0)/(1.0 - exp(-(v+34.0)/10.0))
        b_n(v) = 0.625*exp(-(v+44.0)/80.0)
        a_h(v) = 0.35*exp(-(v+58.0)/20.0)
        b_h(v) = 5.0/(1.0 + exp(-(v+28.0)/10.0))

        m∞(v) = a_m(v)/(a_m(v) + b_m(v))
        eqs = [D(V) ~ g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext,
               D(n) ~ (a_n(V)*(1.0 - n) - b_n(V)*n),
               D(h) ~ (a_h(V)*(1.0 - h) - b_h(V)*h)
        ]
        sys = ODESystem(eqs, t, sts, p; name=name)
        new(p, sts[1], sts[4], sys, namespace)
        end
end