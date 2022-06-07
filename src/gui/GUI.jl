module GUI

using Neuroblox

# constants

const ICONPATH = "img/blockicons/"

Base.@kwdef struct NeurobloxConstructorArgumentData
  default_value
  type
  min_value
  max_value
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
  endswith(name, "Blox") ? name[1:end-4] : name
end

function icon(::Type{T}) where T
  string(ICONPATH, label(T), ".svg")
end

function inputs(::Type{T}) where T
  [Dict(:name => "Default IN")]
end

function outputs(::Type{T}) where T
  [Dict(:name => "Default OUT")]
end

# methods

const NUMBER = "number"
const STRING = "string"

function arguments(::Type{Neuroblox.HarmonicOscillatorBlox})
  Dict(
    :ω => NCAD(0.001, NUMBER, 0.001, 5.0),
    :ζ => NCAD(0.0, NUMBER, 0.0, 500.0),
    :k => NCAD(5.0, NUMBER, 5.0, 700.0),
    :h => NCAD(0.1, NUMBER, 0.1, 5.0)
  )
end

function arguments(::Type{Neuroblox.JansenRitCBlox})
  Dict(
    :τ => NCAD(0.001, NUMBER, 0.001, 5.0),
    :H => NCAD(0.0, NUMBER, 0.0, 500.0),
    :λ => NCAD(5.0, NUMBER, 5.0, 700.0),
    :r => NCAD(0.1, NUMBER, 0.1, 5.0)
  )
end

function arguments(::Type{Neuroblox.JansenRitSCBlox})
  Dict(
    :τ => NCAD(0.001, NUMBER, 0.001, 5.0),
    :H => NCAD(0.0, NUMBER, 0.0, 500.0),
    :λ => NCAD(5.0, NUMBER, 5.0, 700.0),
    :r => NCAD(0.1, NUMBER, 0.1, 5.0)
  )
end

function arguments(::Type{Neuroblox.WilsonCowanBlox})
  Dict(
    :τ_E => NCAD(1.0, NUMBER, 1.0, 100.0),
    :τ_I => NCAD(1.0, NUMBER, 1.0, 100.0),
    :a_E => NCAD(1.2, NUMBER, 1.0, 100.0),
    :a_I => NCAD(2.0, NUMBER, 1.0, 100.0),
    :c_EE => NCAD(5.0, NUMBER, 1.0, 100.0),
    :c_EI => NCAD(10.0, NUMBER, 1.0, 100.0),
    :c_IE => NCAD(6.0, NUMBER, 1.0, 100.0),
    :c_II => NCAD(1.0, NUMBER, 1.0, 100.0),
    :θ_E => NCAD(2.0, NUMBER, 1.0, 100.0),
    :θ_I => NCAD(3.5, NUMBER, 1.0, 100.0),
    :η => NCAD(1.0, NUMBER, 1.0, 100.0)
  )
end

function arguments(::Type{Neuroblox.NextGenerationBlox})
  Dict(
    :C => NCAD(0.0, NUMBER, 0.0, 5.0),
    :Δ => NCAD(0.0, NUMBER, 0.0, 500.0),
    :η_0 => NCAD(5.0, NUMBER, 0.0, 700.0),
    :v_syn => NCAD(0.0, NUMBER, 0.0, 5.0),
    :alpha_inv => NCAD(0.0, NUMBER, 0.0, 5.0),
    :k => NCAD(0.0, NUMBER, 0.0, 5.0)
  )
end

end