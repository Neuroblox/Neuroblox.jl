using Neuroblox, MetaGraphs
import ModelingToolkit: outputs

@named ob1 = BalloonModel()
@named ob2 = BalloonModel()
@named nmm1 = LinearNeuralMass()
@named nmm2 = LinearNeuralMass()

blox = [nmm1, ob1, nmm2, ob2]

g = MetaDiGraph()
add_blox!.(Ref(g), blox)

add_edge!(g, 1, 2, Dict(:weight => 1.0)) # Connect hemodynamic observer #1
add_edge!(g, 3, 4, Dict(:weight => 1.0)) # Connect hemodynamic observer #2
add_edge!(g, 1, 3, Dict(:weight => 1.0)) # Connect neural mass
add_edge!(g, 3, 1, Dict(:weight => 1.0)) # Bidirectional connection because that's the assumption in spectral spectralDCM

@named final_system = system_from_graph(g)

obs_idx, obs_state_names = get_hemodynamic_observers(final_system)

# To use obs_idx
all_outputs = outputs(final_system)
obs_outputs = all_outputs[obs_idx] #equivalent to obs_state_names