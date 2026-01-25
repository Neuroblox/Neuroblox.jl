using OrdinaryDiffEqTsit5
const GD = GraphDynamics
const V_TRACE_TSPAN = (0.0, 1000.0)
const V_TRACE_SAVEAT = 1.0

function with_states(sys, patch)
    tag = GD.get_tag(sys)
    params = NamedTuple(GD.get_params(sys))
    states = merge(NamedTuple(GD.get_states(sys)), patch)
    GD.Subsystem{tag}(; states, params)
end

function max_trace_delta(a, b)
    maximum(abs.(a .- b))
end

function count_spikes(t, v; thresh=0.0, refractory_ms=5.0)
    if length(v) < 2
        return 0
    end
    crossings = findall((v[1:end-1] .<= thresh) .& (v[2:end] .> thresh))
    count = 0
    last_t = -Inf
    for idx in crossings
        t_cross = t[idx+1]
        if t_cross - last_t >= refractory_ms
            count += 1
            last_t = t_cross
        end
    end
    count
end

function solve_trace(g, idxs; tspan=V_TRACE_TSPAN)
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5(); saveat=V_TRACE_SAVEAT, reltol=1e-6, abstol=1e-6)
    sol, sol.t, collect(sol(sol.t, idxs=idxs))
end

function d1_phi_trace(; da=0.0, tspan=V_TRACE_TSPAN)
    @named d1 = MsnD1Receptor()
    @named drive = ConstantDAInput(DA=da)
    g = GraphSystem()
    add_node!(g, d1)
    add_node!(g, drive)
    add_connection!(g, drive, d1; conn=BasicConnection(1.0), weight=1.0)
    solve_trace(g, d1.ϕ1; tspan=tspan)
end

function d2_phi_trace(; da=0.0, tspan=V_TRACE_TSPAN)
    @named d2 = MsnD2Receptor()
    @named drive = ConstantDAInput(DA=da)
    g = GraphSystem()
    add_node!(g, d2)
    add_node!(g, drive)
    add_connection!(g, drive, d2; conn=BasicConnection(1.0), weight=1.0)
    solve_trace(g, d2.ϕ2; tspan=tspan)
end

function nmda_trace(; v_pre=0.0, v_post=0.0, m_nmda1=1.0, tspan=V_TRACE_TSPAN)
    @named nmda = MsnNMDAR()
    @named drive_pre = ConstantVPreInput(V_pre=v_pre)
    @named drive_post = ConstantVPostInput(V_post=v_post)
    @named drive_m = ConstantMNMDA1Input(M_NMDA1=m_nmda1)

    g = GraphSystem()
    add_node!(g, nmda)
    add_node!(g, drive_pre)
    add_node!(g, drive_post)
    add_node!(g, drive_m)
    add_connection!(g, drive_pre, nmda; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_post, nmda; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_m, nmda; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, nmda.A; tspan=tspan)
end

function nmda_current(m_nmda1; v=0.0)
    @named nmda = MsnNMDAR()
    sys_nmda = GD.to_subsystem(nmda)
    sys_nmda = with_states(sys_nmda, (; A=0.1, B=0.5, g=0.2, M_NMDA1_state=m_nmda1))
    props = GD.computed_properties_with_inputs(MsnNMDAR)
    input0 = GD.initialize_input(sys_nmda)
    input = merge(input0, (; V_pre=0.0, V_post=v, M_NMDA1=m_nmda1))
    props.I(sys_nmda, input)
end

function ampa_current(m_ampa; v=-60.0, g_asymp=1.0, tspan=(0.0, 50.0))
    @named syn = MsnAMPAR()
    @named drive_g = ConstantGAsympInput(G_asymp=g_asymp)
    @named drive_m = ConstantMAMPA2Input(M_AMPA2=m_ampa)

    g = GraphSystem()
    add_node!(g, syn)
    add_node!(g, drive_g)
    add_node!(g, drive_m)
    add_connection!(g, drive_g, syn; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_m, syn; conn=BasicConnection(1.0), weight=1.0)

    sol, _, g_trace = solve_trace(g, syn.G; tspan=tspan)
    sys_syn = GD.to_subsystem(syn)
    i_ampa = sys_syn.g * g_trace[end] * (v - sys_syn.E_syn)
    sol, i_ampa
end

function htr5_flags(mode)
    @named selector = HTR5()
    sys = GD.to_subsystem(selector)
    props = GD.computed_properties_with_inputs(HTR5)
    input0 = GD.initialize_input(sys)
    input = merge(input0, (; mode))
    (; PKA=props.PKA(sys, input), PKC=props.PKC(sys, input))
end

function htr5_trace(; mode=0.0, tspan=V_TRACE_TSPAN)
    @named selector = HTR5()
    @named drive = ConstantModeInput(mode=mode)

    g = GraphSystem()
    add_node!(g, selector)
    add_node!(g, drive)
    add_connection!(g, drive, selector; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, selector.dummy; tspan=tspan)
end

function muscarinic_phi_trace(; m=0.0, tspan=V_TRACE_TSPAN)
    @named receptor = MuscarinicR()
    @named drive = ConstantMuscarinicInput(M=m)

    g = GraphSystem()
    add_node!(g, receptor)
    add_node!(g, drive)
    add_connection!(g, drive, receptor; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, receptor.ϕ_M; tspan=tspan)
end

function muscarinic_pair_voltage_trace(; m=0.0, g_ncm=0.01, tspan=V_TRACE_TSPAN)
    @named neuron = MuscarinicNeuron()
    @named receptor = MuscarinicR(ḡ_NCM=g_ncm)
    @named drive = ConstantMuscarinicInput(M=m)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, receptor)
    add_node!(g, drive)
    add_connection!(g, neuron, receptor; weight=1.0)
    add_connection!(g, drive, receptor; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V_m; tspan=tspan)
end

function muscarinic_neuron_voltage_trace(; i_app=0.0, tspan=V_TRACE_TSPAN)
    @named neuron = MuscarinicNeuron()
    @named drive = ConstantIAppInput(I_app=i_app)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V_m; tspan=tspan)
end

function baxter_voltage_trace(; I_stim=0.5, tspan=V_TRACE_TSPAN)
    @named neuron = BaxterSensoryNeuron()
    @named stim = ConstantIStimInput(I=I_stim)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, stim)
    add_connection!(g, stim, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

"""
Tonic variant for BaxterSensoryNeuron test.

Rationale:
- The default BaxterSensoryNeuron is phasic due to Ca-dependent adaptation.
- To elicit multiple spikes under constant drive (1000 ms), we disable the
  slow Ca-dependent K currents (KCaS) by setting gKCaS_00/10/01/11 = 0.
- We keep the same input current units (nA) and run the same solver settings
  to make results comparable to the non-tonic baseline.
"""
function baxter_tonic_voltage_trace(; I_stim=1.0, tspan=V_TRACE_TSPAN)
    @named neuron = BaxterSensoryNeuron(
        gKCaS_00=0.0,
        gKCaS_10=0.0,
        gKCaS_01=0.0,
        gKCaS_11=0.0
    )
    @named stim = ConstantIStimInput(I=I_stim)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, stim)
    add_connection!(g, stim, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

function trn_voltage_trace(; i_app=0.0, tspan=V_TRACE_TSPAN)
    @named neuron = TRNNeuron()
    @named drive = ConstantIAppInput(I_app=i_app)
    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V_m; tspan=tspan)
end

"""
Tonic variant for TRNNeuron test.

Rationale:
- The default TRNNeuron is phasic because Ca-dependent adaptation quickly
  suppresses repetitive firing under constant I_app.
- We reduce adaptation by setting g_KCa = 0 and lowering g_CAN to 0.05.
- This keeps the TRN model intact while producing a tonic spike train in
  a 1000 ms window for the test.
"""
function trn_tonic_voltage_trace(; i_app=6.0, tspan=V_TRACE_TSPAN)
    @named neuron = TRNNeuron(g_KCa=0.0, g_CAN=0.05)
    @named drive = ConstantIAppInput(I_app=i_app)
    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V_m; tspan=tspan)
end

function alpha7_trn_voltage_trace(; ach=0.0, tspan=V_TRACE_TSPAN)
    @named neuron = TRNNeuron()
    @named receptor = Alpha7ERnAChR()
    @named drive_ach = ConstantAChInput(ACh=ach)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, receptor)
    add_node!(g, drive_ach)
    add_connection!(g, drive_ach, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, neuron, receptor; weight=1.0)

    solve_trace(g, neuron.V_m; tspan=tspan)
end

function catrpm4_trn_voltage_trace(; cch=0.0, i_ca=-0.05, ca_bulk=0.05, tspan=V_TRACE_TSPAN)
    @named neuron = TRNNeuron()
    @named receptor = CaTRPM4R()
    @named drive_ica = ConstantICaInput(I_Ca=i_ca)
    @named drive_cab = ConstantCaBulkInput(Ca_bulk=ca_bulk)
    @named drive_cch = ConstantCChInput(CCh=cch)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, receptor)
    add_node!(g, drive_ica)
    add_node!(g, drive_cab)
    add_node!(g, drive_cch)
    add_connection!(g, drive_ica, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_cab, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_cch, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, neuron, receptor; weight=1.0)

    solve_trace(g, neuron.V_m; tspan=tspan)
end

# Beta2nAChR test helpers
function beta2nachr_activation_trace(; inp_ACh=0.0, inp_Nic=0.0, tspan=V_TRACE_TSPAN)
    @named receptor = Beta2nAChR()
    @named drive_ach = ConstantAChInput(ACh=inp_ACh)
    @named drive_nic = ConstantNicInput(Nic=inp_Nic)

    g = GraphSystem()
    add_node!(g, receptor)
    add_node!(g, drive_ach)
    add_node!(g, drive_nic)
    add_connection!(g, drive_ach, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_nic, receptor; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, receptor.ACh_act; tspan=tspan)
end

function beta2nachr_des_trace(; inp_ACh=0.0, inp_Nic=0.0, tspan=V_TRACE_TSPAN)
    @named receptor = Beta2nAChR()
    @named drive_ach = ConstantAChInput(ACh=inp_ACh)
    @named drive_nic = ConstantNicInput(Nic=inp_Nic)

    g = GraphSystem()
    add_node!(g, receptor)
    add_node!(g, drive_ach)
    add_node!(g, drive_nic)
    add_connection!(g, drive_ach, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_nic, receptor; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, receptor.ACh_des; tspan=tspan)
end

function beta2nachr_neuron_voltage_trace(; inp_ACh=0.0, inp_Nic=0.0, g_ACh=5.0, tspan=V_TRACE_TSPAN)
    @named neuron = VTADANeuron()
    @named receptor = Beta2nAChR(ḡ_ACh=g_ACh)
    @named drive_ach = ConstantAChInput(ACh=inp_ACh)
    @named drive_nic = ConstantNicInput(Nic=inp_Nic)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, receptor)
    add_node!(g, drive_ach)
    add_node!(g, drive_nic)
    add_connection!(g, drive_ach, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, drive_nic, receptor; conn=BasicConnection(1.0), weight=1.0)
    add_connection!(g, neuron, receptor; weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

function vtada_voltage_trace(; i_app=0.0, tspan=V_TRACE_TSPAN)
    @named neuron = VTADANeuron()
    @named drive = ConstantIAppInput(I_app=i_app)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

function vtagaba_voltage_trace(; i_app=0.0, tspan=V_TRACE_TSPAN)
    @named neuron = VTAGABANeuron()
    @named drive = ConstantIAppInput(I_app=i_app)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

"""
Tonic spiking variant for VTADANeuron test.

VTADANeuron is a standard HH-type neuron.
With sufficient current (I_app=10.0), it produces tonic spiking.
"""
function vtada_tonic_voltage_trace(; i_app=10.0, tspan=V_TRACE_TSPAN)
    @named neuron = VTADANeuron()
    @named drive = ConstantIAppInput(I_app=i_app)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

"""
Tonic spiking variant for VTAGABANeuron test.

VTAGABANeuron has identical HH dynamics to VTADANeuron.
With sufficient current (I_app=10.0), it produces tonic spiking.
"""
function vtagaba_tonic_voltage_trace(; i_app=10.0, tspan=V_TRACE_TSPAN)
    @named neuron = VTAGABANeuron()
    @named drive = ConstantIAppInput(I_app=i_app)

    g = GraphSystem()
    add_node!(g, neuron)
    add_node!(g, drive)
    add_connection!(g, drive, neuron; conn=BasicConnection(1.0), weight=1.0)

    solve_trace(g, neuron.V; tspan=tspan)
end

