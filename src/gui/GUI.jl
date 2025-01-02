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

function param_order(::Type{T}) where T
  return [k for k in keys(arguments(T))]
end

function plotdetail(::Type{T}) where T
  return OrderedDict()
end

# methods

const NUMBER = "number"
const STRING = "string"
const INTEGER = "integer"
const DROPDOWN = "dropdown"
const NODE = "node"
const LINK = "link"
const FILE = "file"

# GUI tabs

function tab_parameters()
  OrderedDict(
    "Nodes & Edges" => OrderedDict(:order => [],),
    "Params" => OrderedDict(:order => [],),
    "Sim" => OrderedDict(
      :duration => NCAD(600, NUMBER, 1, 100000,[],true),
      :reltol => NCAD(1e-3, NUMBER, 1e-1, 1e-7,[],true),
      :abstol => NCAD(1e-6, NUMBER, 1e-1, 1e-10,[],true),
      :solver => NCAD(1, DROPDOWN, 1, 2,["stiff","non-stiff"],true),
      :dt => NCAD(0.1, NUMBER, 1e-3, 10,[],true),
      :spike_threshold => NCAD(20, NUMBER, -50, 200,[],true),
      :max_neurons => NCAD(10, INTEGER, 1, 100,[],true),
      :neuron_type => NCAD(1, DROPDOWN, 1, 3,["excit","inhib","all"],true),
      :order => ["duration","reltol","abstol","solver","dt","spike_threshold","max_neurons","neuron_type"]
    ),
    "Parameter Fitting" => OrderedDict(
      :method => NCAD(1, DROPDOWN, 1, 2,["Laplace","MCMC"],true),
      :experiment => NCAD(1, DROPDOWN, 1, 2,["fMRI","LFP","EEG"],true),
      :max_iter => NCAD(100, INTEGER, 1, 10000,[],true),
      :accuracy => NCAD(0.05, NUMBER, 1e-3, 10,[],true),
      :ExpData => NCAD("", FILE, 0, 0,["ExpData"],true),
      :order => ["method","experiment","max_iter","accuracy","ExpData"]
    ),
    "Reinforcement Learning" => OrderedDict(
      :trials => NCAD(20, INTEGER, 1, 10000,[],true),
      :t_warmup => NCAD(200, NUMBER, 0, 10000, [], true),
      :order => ["trials","t_warmup"],
    ),
    "Data Loader" => OrderedDict(:order => [],),
  )
end

# Blox arguments and interface
function arguments(::Type{Neuroblox.ImageStimulus})
  OrderedDict(
    :height => NCAD(15, INTEGER, 1, 100,[], false),
    :width => NCAD(15, INTEGER, 1, 100,[], false),
    :N_stims => NCAD(20, INTEGER, 1, 1000, [], false),
    :file => NCAD("", FILE, 0, 0,["Images"], false),
    :t_stimulus => NCAD(700, NUMBER, 10, 10000,[],true),
    :t_pause => NCAD(300, NUMBER, 10, 10000,[],true)
  )
end

function param_order(::Type{Neuroblox.ImageStimulus})
  [:height, :width, :N_stims, :file, :t_stimulus, :t_pause]
end

function arguments(::Type{Neuroblox.WinnerTakeAllBlox})
  OrderedDict(
    :N_exci => NCAD(5, INTEGER, 1, 40, [],true),
    :E_syn_exci => NCAD(0.0, NUMBER, -100, 100.0,[],true),
    :G_syn_exci => NCAD(3.0, NUMBER, -100.0, 100.0,[],true),
    :phase => NCAD(0, NUMBER, 0.0, 2*π,[],true),
	  :τ_exci => NCAD(5, NUMBER, 0.01, 20.0,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -100, 100.0,[],true),
    :G_syn_inhib => NCAD(5.0, NUMBER, 0.01, 20.0,[],true),
	  :τ_inhib => NCAD(70, NUMBER, 0.01, 200,[],true),
    :freq => NCAD(0.0, NUMBER, 0.0, 200, [], true)
  )
end

function inputs(::Type{Neuroblox.WinnerTakeAllBlox})
  ["in1","in2","in3","in4","in5"]
end

function outputs(::Type{Neuroblox.WinnerTakeAllBlox})
  ["out1","out2","out3","out4","out5"]
end

function arguments(::Type{Neuroblox.HarmonicOscillator})
  OrderedDict(
    #:measurement => NCAD("Measurement", MENU, 1 , 4 ,["none","fMRI", "EEG", "LFP"]),
    # MENU NCAD("title of menue", MENU, default, #ofoptions, list of options)
    :ω => NCAD(25*(2*pi)/1000, NUMBER, (2*pi)/1000, 150*(2*pi)/1000,[],true),
    :ζ => NCAD(1.0, NUMBER, -1.0, 1.0,[],true),
    :k => NCAD(625*(2*pi)/1000, NUMBER, (2*pi), 22500*(2*pi)/1000,[],true),
    :h => NCAD(35.0, NUMBER, 0.01, 90.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.HarmonicOscillator})
  OrderedDict(:detail => ["x","y"], :nodetail => ["x"])
end

function arguments(::Type{Neuroblox.JansenRit})
  OrderedDict(
    :τ => NCAD(1, NUMBER, 1, 14,[],true),
    :H => NCAD(20.0, NUMBER, 0.0, 0.5,[],true),
    :λ => NCAD(400.0, NUMBER, 20.0, 500.0,[],true),
    :r => NCAD(0.1, NUMBER, 0.1, 5.0,[],true)
  )
end

function plotdetail(::Type{Neuroblox.JansenRit})
  OrderedDict(:detail => ["x","y"], :nodetail => ["x"])
end

function arguments(::Type{Neuroblox.LinearNeuralMass})
  OrderedDict(
  )
end

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

function arguments(::Type{Neuroblox.Thalamus})
  OrderedDict(
    :N_exci => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_exci => NCAD(0, NUMBER, -70, 70,[],true),
    :G_syn_exci => NCAD(3, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(5, NUMBER, 1, 200,[],true)
  )
end

function plotdetail(::Type{Neuroblox.Thalamus})
  OrderedDict(:mean => "V", :detail => ["V"], :plots =>["raster","stack","adj"])
end

function arguments(::Type{Neuroblox.Striatum})
  OrderedDict(
    :N_inhib => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -20, -100,[],true),
    :G_syn_inhib => NCAD(1.2, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(70, NUMBER, 1, 200,[],true)
  )
end

function plotdetail(::Type{Neuroblox.Striatum})
  OrderedDict(:mean => "V", :detail => ["V"], :plots =>["raster","stack","adj"])
end

function arguments(::Type{Neuroblox.GPe})
  OrderedDict(
    :N_inhib => NCAD(15, INTEGER, 1, 100,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -20, -100,[],true),
    :G_syn_inhib => NCAD(1.2, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(70, NUMBER, 1, 200,[],true)
  )
end

function plotdetail(::Type{Neuroblox.GPe})
  OrderedDict(:mean => "V", :detail => ["V"], :plots =>["raster","stack","adj"])
end

function arguments(::Type{Neuroblox.GPi})
  OrderedDict(
    :N_inhib => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_inhib => NCAD(-70, NUMBER, -20, -100,[],true),
    :G_syn_inhib => NCAD(1.2, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(70, NUMBER, 1, 200,[],true)
  )
end

function plotdetail(::Type{Neuroblox.GPi})
  OrderedDict(:mean => "V", :detail => ["V"], :plots =>["raster","stack","adj"])
end

function arguments(::Type{Neuroblox.STN})
  OrderedDict(
    :N_exci => NCAD(25, INTEGER, 1, 100,[],true),
    :E_syn_exci => NCAD(0, NUMBER, -70, 70,[],true),
    :G_syn_exci => NCAD(3, NUMBER, 0, 10,[],true),
    :τ_inhib => NCAD(5, NUMBER, 1, 200,[],true)
  )
end

function plotdetail(::Type{Neuroblox.STN})
  OrderedDict(:mean => "V", :detail => ["V"], :plots =>["raster","stack","adj"])
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

function arguments(::Type{Neuroblox.TAN})
  OrderedDict(
    :κ => NCAD(100, NUMBER, 0.1, 200,[],true),
    :λ => NCAD(1, NUMBER, 0.1, 2,[],true)
  )
end

function plotdetail(::Type{Neuroblox.TAN})
  OrderedDict(:mean => "R", :detail => ["R"])
end

function arguments(::Type{Neuroblox.HHNeuronExciBlox}) #TODO: add correct settings for the arguments
  OrderedDict(
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

function plotdetail(::Type{Neuroblox.QIFNeuron})
  OrderedDict(:detail => ["V","G","z"], :nodetail => ["V"])
end

function arguments(::Type{Neuroblox.QIFNeuron}) #TODO: add correct settings for the arguments
  OrderedDict(
    :C => NCAD(1.0, NUMBER, 0.1, 100.0,[],true),
    :ω => NCAD(0.0, NUMBER, 0.0, 100.0,[],true),
    :E_syn => NCAD(0.0, NUMBER, -10, 10,[],true),
    :G_syn => NCAD(1.0, NUMBER, 0.1, 5,[],true),
    :τ₁ => NCAD(10.0, NUMBER, 1, 100,[],true),
    :τ₂ => NCAD(10.0, NUMBER, 1, 100,[],true),
    :I_in  => NCAD(0.0, NUMBER, -25, 25,[],true),
    :Eₘ => NCAD(0.0, NUMBER, -10, 10,[],true),
    :Vᵣₑₛ => NCAD(-70, NUMBER, -100, -55,[],true),
	  :θ => NCAD(25, NUMBER, 0, 50,[],true)
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
  OrderedDict(:mean => "V", :detail => ["V"], :plots =>["raster","stack","adj"])
end

# function arguments(::Type{Neuroblox.BandPassFilterBlox})
#   OrderedDict(
#     :lb => NCAD(10, NUMBER, 0, 500,[],true),
#     :ub => NCAD(10, NUMBER, 0, 500,[],true),
#     :fs => NCAD(1000, NUMBER, 1, 10000,[],true),
#     :order => NCAD(4, INTEGER, 1, 2000,[],true)
#   )
# end

# #function arguments(::Type{Neuroblox.PowerSpectrumBlox})
#   OrderedDict(
#     :fs => NCAD(1000, NUMBER, 1, 10000,[],true),
#   )
# end

# function arguments(::Type{Neuroblox.PhaseAngleBlox})
#   OrderedDict(
#   )
# end

end