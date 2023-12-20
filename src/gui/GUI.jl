module GUI

using Neuroblox, OrderedCollections

# constants

const ICONPATH = "img/blockicons/"

Base.@kwdef struct NeurobloxConstructorArgumentData
  default_value
  type
  min_value
  max_value
  menuitems
  isparam
end
const NCAD = NeurobloxConstructorArgumentData

# Abstract functions

function arguments end
function label end
function icon end
function inputs end
function outputs end

# Default functions

function label(::Type{T}) where T
  name = split(string(T), '.'; keepempty = false)[end]
  if endswith(name, "Blox")
    name[1:end-4]
  elseif endswith(name, "Utility")
    name[1:end-7]
  else
    name
  end
end

function icon(::Type{T}) where T
  string(ICONPATH, label(T), ".svg")
end

# the standard neural mass blox has one input and one output
# please define your specific inputs and outputs when the blox
# is different
function inputs(::Type{T}) where T
  ["in"]
end

function outputs(::Type{T}) where T
  ["out"]
end

# methods

const NUMBER = "number"
const STRING = "string"
const INTEGER = "integer"
const MENU = "menu"
const NODE = "node"
const LINK = "link"

# function arguments(::Type{Neuroblox.Merger})
#   OrderedDict(
#   )
# end

# function arguments(::Type{Neuroblox.LinHemo})
#   OrderedDict(
#   )
# end

# function arguments(::Type{Neuroblox.Hemodynamics})
#   OrderedDict(
#   )
# end

function arguments(::Type{Neuroblox.ImageStimulus})
  OrderedDict(
    :height => NCAD(15, INTEGER, 1, 100,[], false),
    :width => NCAD(15, INTEGER, 1, 100,[], false),
    :t_stimulus => NCAD(40, NUMBER, 0.1, 10,[],true),
    :t_pause => NCAD(10, NUMBER, 0.1, 10,[],true)
  )
end

function arguments(::Type{Neuroblox.WinnerTakeAllBlox})
  OrderedDict(
    # :BlockSize => NCAD(5, INTEGER, 1, 40, []),
    :E_syn_exci => NCAD(1.0, NUMBER, 0.01, 100.0,[],true),
    :G_syn_exci => NCAD(5.0, NUMBER, 0.01, 20.0,[],true),
    :I_in => NCAD(0.0, NUMBER, 0.01, 20.0,[],true),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[],true),
	  :τ_exci => NCAD(0.105, NUMBER, 0.01, 2.0,[],true),
    :E_syn_inhib => NCAD(1.0, NUMBER, 0.01, 100.0,[],true),
    :G_syn_inhib => NCAD(5.0, NUMBER, 0.01, 20.0,[],true),
    :I_in => NCAD(0.0, NUMBER, 0.01, 20.0,[],true),
	  :τ_inhib => NCAD(0.105, NUMBER, 0.01, 2.0,[],true)
  )
end

function inputs(::Type{Neuroblox.WinnerTakeAllBlox})
  ["in1","in2","in3","in4","in5"]
end

function outputs(::Type{Neuroblox.WinnerTakeAllBlox})
  ["out1","out2","out3","out4","out5"]
end

# function arguments(::Type{Neuroblox.LinearNeuralMassBlox})
#   OrderedDict(
#     :τ => NCAD(0.01, NUMBER, 1.0, 10.0,[]),
#   )
# end

# function arguments(::Type{Neuroblox.LinearNeuralMass})
#   OrderedDict(
#     :τ => NCAD(0.01, NUMBER, 1.0, 10.0,[]),
#   )
# end

#TODO: create connector compatible stochastic blox
# function arguments(::Type{Neuroblox.OUBlox})
#   OrderedDict(
#     :μ => NCAD(0.0, NUMBER, -100.0, 100.0,[]),
#     :τ => NCAD(1.0, NUMBER, 0.00001,10,[]),
#     :σ => NCAD(1.0, NUMBER, 0.00001,10,[])
#   )
# end

# function arguments(::Type{Neuroblox.OUCouplingBlox})
#   OrderedDict(
#     :μ => NCAD(0.0, NUMBER, -100.0, 100.0,[]),
#     :τ => NCAD(1.0, NUMBER, 0.00001,10,[]),
#     :σ => NCAD(1.0, NUMBER, 0.00001,10,[])
#   )
# end

# function arguments(::Type{Neuroblox.HarmonicOscillatorBlox})
#   OrderedDict(
#     :ω => NCAD(25*(2*pi), NUMBER, (2*pi), 150*(2*pi),[]),
#     :ζ => NCAD(1.0, NUMBER, -1.0, 1.0,[]),
#     :k => NCAD(625*(2*pi), NUMBER, (2*pi), 22500*(2*pi),[]),
#     :h => NCAD(35.0, NUMBER, 0.01, 90.0,[])
#   )
# end

function arguments(::Type{Neuroblox.HarmonicOscillator})
  OrderedDict(
    #:measurement => NCAD("Measurement", MENU, 1 , 4 ,["none","fMRI", "EEG", "LFP"]),
    # MENU NCAD("title of menue", MENU, default, #ofoptions, list of options)
    :ω => NCAD(25*(2*pi), NUMBER, (2*pi), 150*(2*pi),[],true),
    :ζ => NCAD(1.0, NUMBER, -1.0, 1.0,[],true),
    :k => NCAD(625*(2*pi), NUMBER, (2*pi), 22500*(2*pi),[],true),
    :h => NCAD(35.0, NUMBER, 0.01, 90.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.HarmonicOscillator})
  OrderedDict(:detail => ["x","y"], :nodetail => ["x"])
end

# function arguments(::Type{Neuroblox.JansenRitCBlox})
#   OrderedDict(
#     :τ => NCAD(0.001, NUMBER, 0.001, 2.0,[]),
#     :H => NCAD(20.0, NUMBER, 0.0, 500.0,[]),
#     :λ => NCAD(5.0, NUMBER, 1.0, 25.0,[]),
#     :r => NCAD(0.15, NUMBER, 0.1, 5.0,[])
#   )
# end

# function arguments(::Type{Neuroblox.JansenRitSCBlox})
#   OrderedDict(
#     :τ => NCAD(0.014, NUMBER, 0.001, 0.1,[]),
#     :H => NCAD(20.0, NUMBER, 0.0, 500.0,[]),
#     :λ => NCAD(400.0, NUMBER, 20.0, 500.0,[]),
#     :r => NCAD(0.1, NUMBER, 0.1, 5.0,[])
#   )
# end

function arguments(::Type{Neuroblox.JansenRit})
  OrderedDict(
    :τ => NCAD(0.014, NUMBER, 0.001, 0.1,[],true),
    :H => NCAD(20.0, NUMBER, 0.0, 500.0,[],true),
    :λ => NCAD(400.0, NUMBER, 20.0, 500.0,[],true),
    :r => NCAD(0.1, NUMBER, 0.1, 5.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.JansenRit})
  OrderedDict(:detail => ["x","y"], :nodetail => ["x"])
end

# function arguments(::Type{Neuroblox.WilsonCowanBlox})
#   OrderedDict(
#     :τ_E => NCAD(1.0, NUMBER, 1.0, 100.0,[]),
#     :τ_I => NCAD(1.0, NUMBER, 1.0, 100.0,[]),
#     :a_E => NCAD(1.2, NUMBER, 1.0, 100.0,[]),
#     :a_I => NCAD(2.0, NUMBER, 1.0, 100.0,[]),
#     :c_EE => NCAD(5.0, NUMBER, 1.0, 100.0,[]),
#     :c_EI => NCAD(10.0, NUMBER, 1.0, 100.0,[]),
#     :c_IE => NCAD(6.0, NUMBER, 1.0, 100.0,[]),
#     :c_II => NCAD(1.0, NUMBER, 1.0, 100.0,[]),
#     :θ_E => NCAD(2.0, NUMBER, 1.0, 100.0,[]),
#     :θ_I => NCAD(3.5, NUMBER, 1.0, 100.0,[]),
#     :η => NCAD(1.0, NUMBER, 1.0, 100.0,[])
#   )
# end

function arguments(::Type{Neuroblox.WilsonCowan})
  OrderedDict(
    :τ_E => NCAD(1.0, NUMBER, 1.0, 100.0,[],true),
    :τ_I => NCAD(1.0, NUMBER, 1.0, 100.0,[],true),
    :a_E => NCAD(1.2, NUMBER, 1.0, 100.0,[],true),
    :a_I => NCAD(2.0, NUMBER, 1.0, 100.0,[],true),
    :c_EE => NCAD(5.0, NUMBER, 1.0, 100.0,[],true),
    :c_EI => NCAD(10.0, NUMBER, 1.0, 100.0,[],true),
    :c_IE => NCAD(6.0, NUMBER, 1.0, 100.0,[],true),
    :c_II => NCAD(1.0, NUMBER, 1.0, 100.0,[],true),
    :θ_E => NCAD(2.0, NUMBER, 1.0, 100.0,[],true),
    :θ_I => NCAD(3.5, NUMBER, 1.0, 100.0,[],true),
    :η => NCAD(1.0, NUMBER, 1.0, 100.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.WilsonCowan})
  OrderedDict(:detail => ["E","I"], :nodetail => ["E"])
end

# function arguments(::Type{Neuroblox.LarterBreakspearBlox})
#   OrderedDict(
#     :C => NCAD(0.35, NUMBER, 0.0, 1.0,[]),
#     :δ_VZ => NCAD(0.61, NUMBER, 0.1, 2.0,[]),
#     :T_Ca => NCAD(-0.01, NUMBER, 0.02, -0.04,[]),
#     :δ_Ca => NCAD(0.15, NUMBER, 0.1, 0.2,[]),
#     :g_Ca => NCAD(1.0, NUMBER, 0.96, 1.01,[]), #tested in Jolien's work/similar to V_Ca in Anthony's paper
#     :V_Ca => NCAD(1.0, NUMBER, 0.96, 1.01,[]), #limits established by bifurcation
#     :T_K => NCAD(0.0, NUMBER, -0.05, 0.05,[]),
#     :δ_K => NCAD(0.3, NUMBER,0.25, 0.35,[]),
#     :g_K => NCAD(2.0, NUMBER, 1.95, 2.05,[]),  #tested in Jolien's work
#     :V_K => NCAD(-0.7, NUMBER, -0.8, -0.6,[]),  #limits established by bifurcation
#     :T_Na => NCAD(0.3, NUMBER, 0.25, 0.35,[]),
#     :δ_Na => NCAD(0.15, NUMBER, 0.1, 0.2,[]),
#     :g_Na => NCAD(6.7, NUMBER, 6.6, 6.8,[]),   #tested in Botond and Jolien's work
#     :V_Na => NCAD(0.53, NUMBER, 0.41, 0.59,[]), #limits established by bifurcation
#     :V_L => NCAD(-0.5, NUMBER, -0.6, -0.4,[]),
#     :g_L => NCAD(0.5, NUMBER, 0.4, 0.6,[]),
#     :V_T => NCAD(0.0, NUMBER, -0.05, 0.05,[]),
#     :Z_T => NCAD(0.0, NUMBER, -0.05, 0.05,[]),
#     :IS => NCAD(0.3, NUMBER, 0.0, 1.0,[]),
#     :a_ee => NCAD(0.36, NUMBER, 0.33, 0.39,[]), #tested in Botond and Jolien's work
#     :a_ei => NCAD(2.0, NUMBER, 1.95, 2.05,[]), #tested in Botond and Jolien's work
#     :a_ie => NCAD(2.0, NUMBER, 1.95, 2.05,[]), #testing in Jolien's work
#     :a_ne => NCAD(1.0, NUMBER, 0.95, 1.05,[]),
#     :a_ni => NCAD(0.4, NUMBER, 0.3, 0.5,[]),
#     :b => NCAD(0.1, NUMBER, 0.05, 0.15,[]),
#     :τ_K => NCAD(1.0, NUMBER, 0.8, 1.2,[]), #shouldn't be varied, but useful in bifurcations to "harshen" the potassium landscape
#     :ϕ => NCAD(0.7, NUMBER, 0.6, 0.8,[]),
#     :r_NMDA => NCAD( 0.25, NUMBER, 0.2, 0.3,[]) #tested in Botond's work
#   )
# end

function arguments(::Type{Neuroblox.LarterBreakspear})
  OrderedDict(
    :C => NCAD(0.35, NUMBER, 0.0, 1.0,[],true),
    :δ_VZ => NCAD(0.61, NUMBER, 0.1, 2.0,[],true),
    :T_Ca => NCAD(-0.01, NUMBER, 0.02, -0.04,[],true),
    :δ_Ca => NCAD(0.15, NUMBER, 0.1, 0.2,[],true),
    :g_Ca => NCAD(1.0, NUMBER, 0.96, 1.01,[],true), #tested in Jolien's work/similar to V_Ca in Anthony's paper
    :V_Ca => NCAD(1.0, NUMBER, 0.96, 1.01,[],true), #limits established by bifurcation
    :T_K => NCAD(0.0, NUMBER, -0.05, 0.05,[],true),
    :δ_K => NCAD(0.3, NUMBER,0.25, 0.35,[],true),
    :g_K => NCAD(2.0, NUMBER, 1.95, 2.05,[],true),  #tested in Jolien's work
    :V_K => NCAD(-0.7, NUMBER, -0.8, -0.6,[],true),  #limits established by bifurcation
    :T_Na => NCAD(0.3, NUMBER, 0.25, 0.35,[],true),
    :δ_Na => NCAD(0.15, NUMBER, 0.1, 0.2,[],true),
    :g_Na => NCAD(6.7, NUMBER, 6.6, 6.8,[],true),   #tested in Botond and Jolien's work
    :V_Na => NCAD(0.53, NUMBER, 0.41, 0.59,[],true), #limits established by bifurcation
    :V_L => NCAD(-0.5, NUMBER, -0.6, -0.4,[],true),
    :g_L => NCAD(0.5, NUMBER, 0.4, 0.6,[],true),
    :V_T => NCAD(0.0, NUMBER, -0.05, 0.05,[],true),
    :Z_T => NCAD(0.0, NUMBER, -0.05, 0.05,[],true),
    :IS => NCAD(0.3, NUMBER, 0.0, 1.0,[],true),
    :a_ee => NCAD(0.36, NUMBER, 0.33, 0.39,[],true), #tested in Botond and Jolien's work
    :a_ei => NCAD(2.0, NUMBER, 1.95, 2.05,[],true), #tested in Botond and Jolien's work
    :a_ie => NCAD(2.0, NUMBER, 1.95, 2.05,[],true), #testing in Jolien's work
    :a_ne => NCAD(1.0, NUMBER, 0.95, 1.05,[],true),
    :a_ni => NCAD(0.4, NUMBER, 0.3, 0.5,[],true),
    :b => NCAD(0.1, NUMBER, 0.05, 0.15,[],true),
    :τ_K => NCAD(1.0, NUMBER, 0.8, 1.2,[],true), #shouldn't be varied, but useful in bifurcations to "harshen" the potassium landscape
    :ϕ => NCAD(0.7, NUMBER, 0.6, 0.8,[],true),
    :r_NMDA => NCAD( 0.25, NUMBER, 0.2, 0.3,[],true) #tested in Botond's work
  )
end

function plotdetail(::Type{Neuroblox.LarterBreakspear})
  OrderedDict(:detail => ["V","Z","W"], :nodetail => ["V"])
end

# function arguments(::Type{Neuroblox.NextGenerationBlox})
#   OrderedDict(
#     :C => NCAD(30.0, NUMBER, 1.0, 50.0,[]),
#     :Δ => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
#     :η_0 => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
#     :v_syn => NCAD(-10.0, NUMBER, -20.0, 0.0,[]),
#     :alpha_inv => NCAD(35.0, NUMBER, 0.01, 10.0,[]),
#     :k => NCAD(0.105, NUMBER, 0.01, 2.0,[])
#   )
# end

function arguments(::Type{Neuroblox.NextGenerationEIBlox})
  OrderedDict(
    :Cₑ => NCAD(52.0, NUMBER, 1.0, 50.0,[],true),
    :Cᵢ => NCAD(26.0, NUMBER, 1.0, 50.0,[],true),
    :Δₑ => NCAD(0.5, NUMBER, 0.01, 100.0,[],true),
    :Δᵢ => NCAD(0.5, NUMBER, 0.01, 100.0,[],true),
    :η_0ₑ => NCAD(10.0, NUMBER, 0.01, 20.0,[],true),
    :η_0ᵢ => NCAD(0.0, NUMBER, 0.01, 20.0,[],true),
    :v_synₑₑ => NCAD(10.0, NUMBER, -20.0, 20.0,[],true),
    :v_synₑᵢ => NCAD(-10.0, NUMBER, -20.0, 20.0,[],true),
    :v_synᵢₑ => NCAD(10.0, NUMBER, -20.0, 20.0,[],true),
    :v_synᵢᵢ => NCAD(-10.0, NUMBER, -20.0, 20.0,[],true),
    :alpha_invₑₑ => NCAD(10.0/26, NUMBER, 0.01, 20.0,[],true),
    :alpha_invₑᵢ => NCAD(0.8/26, NUMBER, 0.01, 20.0,[],true),
    :alpha_invᵢₑ => NCAD(10.0/26, NUMBER, 0.01, 20.0,[],true),
    :alpha_invᵢᵢ => NCAD(0.8/26, NUMBER, 0.01, 20.0,[],true),
    :kₑₑ => NCAD(0.0, NUMBER, 0.01, 20.0,[],true),
    :kₑᵢ => NCAD(0.6*26, NUMBER, 0.01, 20.0,[],true),
    :kᵢₑ => NCAD(0.6*26, NUMBER, 0.01, 20.0,[],true),
    :kᵢᵢ => NCAD(0.0, NUMBER, 0.01, 20.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.NextGenerationEIBlox})
  OrderedDict(:detail => ["aₑ","bₑ","aᵢ","bᵢ"], :nodetail => ["aₑ","bₑ"])
end

function arguments(::Type{Neuroblox.HebbianModulationPlasticity})
  OrderedDict(
    :K => NCAD(0.2, NUMBER, 0.01, 1.0,[],true),
    :decay => NCAD(0.01, NUMBER, 0.001, 1.0,[],true),
    :modulator => NCAD("", NODE, 1.0, 100.0,[],true),
    :t_pre => NCAD(2.0, NUMBER, 0.1, 10.0,[],true),
    :t_post => NCAD(2.0, NUMBER, 0.1, 10.0,[],true),
    :t_mod => NCAD(0.7, NUMBER, 0.001, 10.0,[],true)
  )
end

function info_link(::Type{Neuroblox.HebbianModulationPlasticity})
  Dict(:link => "https://www.neuroblox.org")
end

function arguments(::Type{Neuroblox.HebbianPlasticity})
  OrderedDict(
    :K => NCAD(0.2, NUMBER, 0.01, 1.0,[],true),
    :W_lim => NCAD(2.0, NUMBER, 0.0, 10.0,[],true),
    :t_pre => NCAD(2.0, NUMBER, 0.1, 10.0,[],true),
    :t_post => NCAD(2.0, NUMBER, 0.1, 10.0,[],true)
  )
end

function info_link(::Type{Neuroblox.HebbianPlasticity})
  Dict(:link => "https://www.neuroblox.org")
end

#TODO: create connector compatible CMC
# function arguments(::Type{Neuroblox.CanonicalMicroCircuitBlox})
#   OrderedDict(
#     :τ_ss => NCAD(0.002, NUMBER, 0.0001, 0.1,[]),
#     :τ_sp => NCAD(0.002, NUMBER, 0.0001, 0.1,[]),
#     :τ_ii => NCAD(0.016, NUMBER, 0.0001, 0.1,[]),
#     :τ_dp => NCAD(0.028, NUMBER, 0.0001, 0.1,[]),
#     :r_ss => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[]),
#     :r_sp => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[]),
#     :r_ii => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[]),
#     :r_dp => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[])
#   )
# end

# function inputs(::Type{Neuroblox.CanonicalMicroCircuitBlox})
#   ["in_ss","in_sp","in_ii","in_sp"]
# end

# function outputs(::Type{Neuroblox.CanonicalMicroCircuitBlox})
#   ["out_ss","out_sp","out_ii","out_sp"]
# end

function arguments(::Type{Neuroblox.Thalamus})
  OrderedDict(
    :N_exci => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_exci => NCAD(0, NUMBER, -70, 70,[],true),
    :G_syn_exci => NCAD(3, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(5, NUMBER, 1, 200,[],true)
  )
end

function arguments(::Type{Neuroblox.Striatum})
  OrderedDict(
    :N_inhib => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -20, -100,[],true),
    :G_syn_inhib => NCAD(1.2, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(70, NUMBER, 1, 200,[],true)
  )
end

function arguments(::Type{Neuroblox.GPe})
  OrderedDict(
    :N_inhib => NCAD(15, INTEGER, 1, 100,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -20, -100,[],true),
    :G_syn_inhib => NCAD(1.2, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(70, NUMBER, 1, 200,[],true)
  )
end

function arguments(::Type{Neuroblox.GPi})
  OrderedDict(
    :N_inhib => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -20, -100,[],true),
    :G_syn_inhib => NCAD(1.2, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(70, NUMBER, 1, 200,[],true)
  )
end

function arguments(::Type{Neuroblox.STN})
  OrderedDict(
    :N_exci => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_exci => NCAD(0, NUMBER, -70, 70,[],true),
    :G_syn_exci => NCAD(3, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(5, NUMBER, 1, 200,[],true)
  )
end

function arguments(::Type{Neuroblox.SNc})
  OrderedDict(
    :κ_DA => NCAD(0.2, NUMBER, 0.1, 2,[],true),
    :N_time_blocks => NCAD(5, INTEGER, 1, 100,[],true),
    :DA_reward => NCAD(10, NUMBER, 0, 100,[],true)
  )
end

function arguments(::Type{Neuroblox.GreedyPolicy})
  OrderedDict(
    :t_decision => NCAD(300.0, NUMBER, 0.1, 1000.0,[],true)
  )
end

#TODO: create connector compatible CMC
# function arguments(::Type{Neuroblox.IFNeuronBlox}) #TODO: add correct settings for the arguments
#   OrderedDict(
#     :C => NCAD(30.0, NUMBER, 1.0, 50.0,[]),
#     :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
#     :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
# 	  :I_in => NCAD(-10.0, NUMBER, -20.0, 0.0,[]),
# 	  :freq => NCAD(35.0, NUMBER, 0.01, 10.0,[]),
# 	  :phase => NCAD(0.105, NUMBER, 0.01, 2.0,[]),
# 	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[])
#   )
# end

# function arguments(::Type{Neuroblox.QIFNeuronBlox}) #TODO: add correct settings for the arguments
#   OrderedDict(
#     :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
#     :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
# 	  :w => NCAD(-10.0, NUMBER, -20.0, 0.0,[]),
# 	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[])
#   )
# end

# function arguments(::Type{Neuroblox.LIFNeuronBlox}) #TODO: add correct settings for the arguments
#   OrderedDict(
#     :I_in => NCAD(0.0, NUMBER, -5.0, 5.0,[]),
#     :V_L => NCAD(-70.0, NUMBER, -90, -10,[]),
#     :τ => NCAD(10.0, NUMBER, 0.01, 100.0,[]),
#     :R => NCAD(100.0, NUMBER, 0.01, 200.0,[]),
#     :θ => NCAD(-10.0, NUMBER, -20, 20.0,[]),
#   )
# end

function arguments(::Type{Neuroblox.HHNeuronExciBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :t_spike_window => NCAD(90.0, NUMBER, 0.1, 100.0,[],true),
    :θ_spike => NCAD(0.0, NUMBER, 0.0, 2*π,[],true),
    :E_syn => NCAD(0.0, NUMBER, 0.01, 100.0,[],true),
    :G_syn => NCAD(3.0, NUMBER, 0.01, 20.0,[],true),
    :I_bg => NCAD(0.0, NUMBER, 0.0, 20.0,[],true),
    :freq => NCAD(0, NUMBER, 0.0, 100,[],true),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[],true),
	  :τ => NCAD(5, NUMBER, 0.01, 20.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.HHNeuronExciBlox})
  OrderedDict(:detail => ["V","n","m","h"], :nodetail => ["V"])
end

function arguments(::Type{Neuroblox.HHNeuronInhibBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[],true),
    :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[],true),
    :I_bg => NCAD(0.0, NUMBER, 0.01, 20.0,[],true),
    :freq => NCAD(20, NUMBER, 0.01, 100,[],true),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[],true),
	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.HHNeuronInhibBlox})
  OrderedDict(:detail => ["V","n","m","h"], :nodetail => ["V"])
end

function arguments(::Type{Neuroblox.CorticalBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :N_wta => NCAD(10, INTEGER, 1, 100, [],true),
    :N_exci => NCAD(5, INTEGER, 1, 100, [],true),
    :E_syn_exci => NCAD(0.0, NUMBER, 0.01, 100.0,[],true),
    :E_syn_inhib => NCAD(1.0, NUMBER, 0.01, 100.0,[],true),
    :G_syn_exci => NCAD(3.0, NUMBER, 0.01, 20.0,[],true),
    :G_syn_inhib => NCAD(5.0, NUMBER, 0.01, 20.0,[],true),
    :G_syn_ff_inhib => NCAD(3.5, NUMBER, 0.01, 20.0,[],true),
    :freq => NCAD(0, NUMBER, 0.0, 100,[],true),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[],true),
    :I_bg_ar => NCAD(0.0, NUMBER, 0.01, 20.0,[],true),
	  :τ_exci => NCAD(5, NUMBER, 0.01, 20.0,[],true),
	  :τ_inhib => NCAD(70, NUMBER, 0.01, 100.0,[],true),
    :density => NCAD(0.1, NUMBER, 0.01, 1.0,[],true),
    :weight => NCAD(1.0, NUMBER, 0.01, 100.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.CorticalBlox})
  OrderedDict(:mean => "V", :detail => ["V"])
end

function arguments(::Type{Neuroblox.BandPassFilterBlox})
  OrderedDict(
    :lb => NCAD(10, NUMBER, 0, 500,[],true),
    :ub => NCAD(10, NUMBER, 0, 500,[],true),
    :fs => NCAD(1000, NUMBER, 1, 10000,[],true),
    :order => NCAD(4, INTEGER, 1, 2000,[],true)
  )
end

function arguments(::Type{Neuroblox.PowerSpectrumBlox})
  OrderedDict(
    :T => NCAD(10, NUMBER, 0, 2000,[],true),
    :fs => NCAD(1000, NUMBER, 1, 10000,[],true),
  )
end

function arguments(::Type{Neuroblox.PhaseAngleBlox})
  OrderedDict(
  )
end

end