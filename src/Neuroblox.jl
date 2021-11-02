module Neuroblox

using Reexport
@reexport using ModelingToolkit

@parameters t
D = Differential(t)

function NeuralMass(;name, τ=τ, H=H, λ=λ, r=r)

       sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
       params = @parameters τ=τ H=H λ=λ r=r

       eqs = [D(x) ~ y - ((2/τ)*x),
              D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]

       return ODESystem(eqs, t, sts, params; name=name)
end

function Connections(;name, sys=sys)

       params =  @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

       adj_matrix = [0 0 0 0 0 0 0 0;
       -0.5*C_BG_Th*sys[1].x -0.5*C_BG_Th*sys[2].x C_BG_Th*sys[3].x 0 0 0 0 0;
                     0 -0.5*C_BG_Th*sys[2].x 0 0 0 0 C_Cor_BG_Th*sys[7].x 0;
                     0 -0.5*C_BG_Th*sys[2].x C_BG_Th*sys[3].x 0 0 0 0 0;
                     0 0 0 -0.5*C_BG_Th*sys[4].x 0 0 0 0;
                     0 0 0 0 C_BG_Th_Cor*sys[5].x 0 6*C_Cor*sys[7].x 0;
                     0 0 0 0 0 4.8*C_Cor*sys[6].x 0 -1.5*C_Cor*sys[8].x;
                     0 0 0 0 0 0 1.5*C_Cor*sys[7].x 3.3*C_Cor*sys[8].x]
        begin
               eqs = []
               for region_num in 1:length(sys)
                      push!(eqs, sys[region_num].jcn ~ sum(adj_matrix[region_num,:]))
               end
        end

        return @named Circuit = ODESystem(eqs, systems = sys)
end

export NeuralMass, Connections

end
