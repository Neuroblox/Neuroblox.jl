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

function inputs(::Type{T}) where T
  String[]
end

function outputs(::Type{T}) where T
  String[]
end

# methods

const NUMBER = "number"
const STRING = "string"
const INTEGER = "integer"
const MENU = "menu"

function arguments(::Type{Neuroblox.Merger})
  OrderedDict(
  )
end

function arguments(::Type{Neuroblox.LinHemo})
  OrderedDict(
  )
end

function arguments(::Type{Neuroblox.Hemodynamics})
  OrderedDict(
  )
end

function arguments(::Type{Neuroblox.WinnerTakeAllBlox})
  OrderedDict(
  )
end

function arguments(::Type{Neuroblox.LinearNeuralMassBlox})
  OrderedDict(
    :τ => NCAD(0.01, NUMBER, 1.0, 10.0,[]),
  )
end

function arguments(::Type{Neuroblox.HarmonicOscillatorBlox})
  OrderedDict(
    :ω => NCAD(25*(2*pi), NUMBER, (2*pi), 150*(2*pi),[]),
    :ζ => NCAD(1.0, NUMBER, -1.0, 1.0,[]),
    :k => NCAD(625*(2*pi), NUMBER, (2*pi), 22500*(2*pi),[]),
    :h => NCAD(35.0, NUMBER, 0.01, 90.0,[])
  )
end

function arguments(::Type{Neuroblox.JansenRitCBlox})
  OrderedDict(
    :τ => NCAD(0.001, NUMBER, 0.001, 2.0,[]),
    :H => NCAD(20.0, NUMBER, 0.0, 500.0,[]),
    :λ => NCAD(5.0, NUMBER, 1.0, 25.0,[]),
    :r => NCAD(0.15, NUMBER, 0.1, 5.0,[])
  )
end

function arguments(::Type{Neuroblox.JansenRitSCBlox})
  OrderedDict(
    :τ => NCAD(0.014, NUMBER, 0.001, 0.1,[]),
    :H => NCAD(20.0, NUMBER, 0.0, 500.0,[]),
    :λ => NCAD(400.0, NUMBER, 20.0, 500.0,[]),
    :r => NCAD(0.1, NUMBER, 0.1, 5.0,[])
  )
end

function arguments(::Type{Neuroblox.WilsonCowanBlox})
  OrderedDict(
    :τ_E => NCAD(1.0, NUMBER, 1.0, 100.0,[]),
    :τ_I => NCAD(1.0, NUMBER, 1.0, 100.0,[]),
    :a_E => NCAD(1.2, NUMBER, 1.0, 100.0,[]),
    :a_I => NCAD(2.0, NUMBER, 1.0, 100.0,[]),
    :c_EE => NCAD(5.0, NUMBER, 1.0, 100.0,[]),
    :c_EI => NCAD(10.0, NUMBER, 1.0, 100.0,[]),
    :c_IE => NCAD(6.0, NUMBER, 1.0, 100.0,[]),
    :c_II => NCAD(1.0, NUMBER, 1.0, 100.0,[]),
    :θ_E => NCAD(2.0, NUMBER, 1.0, 100.0,[]),
    :θ_I => NCAD(3.5, NUMBER, 1.0, 100.0,[]),
    :η => NCAD(1.0, NUMBER, 1.0, 100.0,[])
  )
end

function arguments(::Type{Neuroblox.LarterBreakspearBlox})
  OrderedDict(
    :C => NCAD(0.35, NUMBER, 0.0, 1.0,[]),
    :δ_VZ => NCAD(0.61, NUMBER, 0.1, 2.0,[]),
    :T_Ca => NCAD(-0.01, NUMBER, 0.02, -0.04,[]),
    :δ_Ca => NCAD(0.15, NUMBER, 0.1, 0.2,[]),
    :g_Ca => NCAD(1.0, NUMBER, 0.96, 1.01,[]), #tested in Jolien's work/similar to V_Ca in Anthony's paper
    :V_Ca => NCAD(1.0, NUMBER, 0.96, 1.01,[]), #limits established by bifurcation
    :T_K => NCAD(0.0, NUMBER, -0.05, 0.05,[]),
    :δ_K => NCAD(0.3, NUMBER,0.25, 0.35,[]),
    :g_K => NCAD(2.0, NUMBER, 1.95, 2.05,[]),  #tested in Jolien's work
    :V_K => NCAD(-0.7, NUMBER, -0.8, -0.6,[]),  #limits established by bifurcation
    :T_Na => NCAD(0.3, NUMBER, 0.25, 0.35,[]),
    :δ_Na => NCAD(0.15, NUMBER, 0.1, 0.2,[]),
    :g_Na => NCAD(6.7, NUMBER, 6.6, 6.8,[]),   #tested in Botond and Jolien's work
    :V_Na => NCAD(0.53, NUMBER, 0.41, 0.59,[]), #limits established by bifurcation
    :V_L => NCAD(-0.5, NUMBER, -0.6, -0.4,[]),
    :g_L => NCAD(0.5, NUMBER, 0.4, 0.6,[]),
    :V_T => NCAD(0.0, NUMBER, -0.05, 0.05,[]),
    :Z_T => NCAD(0.0, NUMBER, -0.05, 0.05,[]),
    :IS => NCAD(0.3, NUMBER, 0.0, 1.0,[]),
    :a_ee => NCAD(0.36, NUMBER, 0.33, 0.39,[]), #tested in Botond and Jolien's work
    :a_ei => NCAD(2.0, NUMBER, 1.95, 2.05,[]), #tested in Botond and Jolien's work
    :a_ie => NCAD(2.0, NUMBER, 1.95, 2.05,[]), #testing in Jolien's work
    :a_ne => NCAD(1.0, NUMBER, 0.95, 1.05,[]),
    :a_ni => NCAD(0.4, NUMBER, 0.3, 0.5,[]),
    :b => NCAD(0.1, NUMBER, 0.05, 0.15,[]),
    :τ_K => NCAD(1.0, NUMBER, 0.8, 1.2,[]), #shouldn't be varied, but useful in bifurcations to "harshen" the potassium landscape
    :ϕ => NCAD(0.7, NUMBER, 0.6, 0.8,[]),
    :r_NMDA => NCAD( 0.25, NUMBER, 0.2, 0.3,[]) #tested in Botond's work
  )
end

function arguments(::Type{Neuroblox.NextGenerationBlox})
  OrderedDict(
    :C => NCAD(30.0, NUMBER, 1.0, 50.0,[]),
    :Δ => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
    :η_0 => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
    :v_syn => NCAD(-10.0, NUMBER, -20.0, 0.0,[]),
    :alpha_inv => NCAD(35.0, NUMBER, 0.01, 10.0,[]),
    :k => NCAD(0.105, NUMBER, 0.01, 2.0,[])
  )
end

function arguments(::Type{Neuroblox.CanonicalMicroCircuitBlox})
  OrderedDict(
    :τ_ss => NCAD(0.002, NUMBER, 0.0001, 0.1,[]),
    :τ_sp => NCAD(0.002, NUMBER, 0.0001, 0.1,[]),
    :τ_ii => NCAD(0.016, NUMBER, 0.0001, 0.1,[]),
    :τ_dp => NCAD(0.028, NUMBER, 0.0001, 0.1,[]),
    :r_ss => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[]),
    :r_sp => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[]),
    :r_ii => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[]),
    :r_dp => NCAD(2.0/3.0, NUMBER, 0.1, 5.0,[])
  )
end

function inputs(::Type{Neuroblox.CanonicalMicroCircuitBlox})
  ["in1","in2","in3","in4"]
end

function outputs(::Type{Neuroblox.CanonicalMicroCircuitBlox})
  ["out1","out2","out3","out4"]
end

function arguments(::Type{Neuroblox.IFNeuronBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :C => NCAD(30.0, NUMBER, 1.0, 50.0,[]),
    :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
    :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
	  :I_in => NCAD(-10.0, NUMBER, -20.0, 0.0,[]),
	  :freq => NCAD(35.0, NUMBER, 0.01, 10.0,[]),
	  :phase => NCAD(0.105, NUMBER, 0.01, 2.0,[]),
	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[])
  )
end

function arguments(::Type{Neuroblox.QIFNeuronBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
    :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
	  :w => NCAD(-10.0, NUMBER, -20.0, 0.0,[]),
	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[])
  )
end

function arguments(::Type{Neuroblox.LIFNeuronBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :I_in => NCAD(0.0, NUMBER, -5.0, 5.0,[]),
    :V_L => NCAD(-70.0, NUMBER, -90, -10,[]),
    :τ => NCAD(10.0, NUMBER, 0.01, 100.0,[]),
    :R => NCAD(100.0, NUMBER, 0.01, 200.0,[]),
    :θ => NCAD(-10.0, NUMBER, -20, 20.0,[]),
  )
end

function arguments(::Type{Neuroblox.HHNeuronExciBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
    :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
    :I_in => NCAD(0.0, NUMBER, 0.01, 20.0,[]),
    :freq => NCAD(20, NUMBER, 0.01, 100,[]),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[]),
	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[])
  )
end

function arguments(::Type{Neuroblox.HHNeuronInhibBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :E_syn => NCAD(1.0, NUMBER, 0.01, 100.0,[]),
    :G_syn => NCAD(5.0, NUMBER, 0.01, 20.0,[]),
    :I_in => NCAD(0.0, NUMBER, 0.01, 20.0,[]),
    :freq => NCAD(20, NUMBER, 0.01, 100,[]),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[]),
	  :τ => NCAD(0.105, NUMBER, 0.01, 2.0,[])
  )
end

function arguments(::Type{Neuroblox.CorticalBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
    :nblocks => NCAD(10, INTEGER, 1, 40, []),
    :blocksize => NCAD(6, INTEGER, 1, 40, []),
    :lfp => NCAD("Average", MENU,1 ,3 ,["Average", "Gaussian", "Peaked"])
  )
end

function arguments(::Type{Neuroblox.BandPassFilterBlox})
  OrderedDict(
    :lb => NCAD(10, NUMBER, 0, 500,[]),
    :ub => NCAD(10, NUMBER, 0, 500,[]),
    :fs => NCAD(1000, NUMBER, 1, 10000,[]),
    :order => NCAD(4, INTEGER, 1, 2000,[])
  )
end

function arguments(::Type{Neuroblox.PowerSpectrumBlox})
  OrderedDict(
    :T => NCAD(10, NUMBER, 0, 2000,[]),
    :fs => NCAD(1000, NUMBER, 1, 10000,[]),
  )
end

function arguments(::Type{Neuroblox.PhaseAngleBlox})
  Dict(
  )
end

end