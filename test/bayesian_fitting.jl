using Neuroblox, StochasticDiffEq ,ForwardDiff, Test
using Turing
using Distributions
using Turing: Variational
using Random
using Tracker

@named VdP = van_der_pol()

prob_vdp = SDEProblem(VdP,[0.1,0.1],[0.0, 10.0],[])
sol = solve(prob_vdp,EM(),dt=0.1)
time = sol.t
dt = time[2]-time[1]
x = sol[1,:]
y = sol[2,:]

# turing model for fitting SDE without external noise
@model function fitvpEM(datax, datay)
  θ ~ Gamma(0.1,10.0)
  ϕ ~ Uniform(0,0.5)
  dxh = datay[1:end-1]
  dyh = θ .* (1 .- datax[1:end-1] .^ 2) .* datay[1:end-1] .- datax[1:end-1]
  datax[2:end] ~ MvNormal(datax[1:end-1] .+ dxh * dt,ϕ*sqrt(dt))
  datay[2:end] ~ MvNormal(datay[1:end-1] .+ dyh * dt,ϕ*sqrt(dt))
end

Turing.setadbackend(:forwarddiff)
modelEM = fitvpEM(x,y)
chain = Turing.sample(modelEM,NUTS(0.65),200)
@test 0.5<mean(chain[:θ])<1.5
@test 0.05<mean(chain[:ϕ])<0.15

#turing model for fitting with external noise
noise = Normal(0,0.3)
xn = x .+ rand(noise,length(x))
yn = y .+ rand(noise,length(y))

@model function fitvpEMn(datax, datay)
  σ = 0.3 # here we explicitly set the noise, but it could also be a parameter
  θ ~ Gamma(0.25,4.0)
  ϕ ~ Uniform(0,0.2)
  # create arrays for hidden variables
  xh = eltype(θ).(zero(datax))
  yh = eltype(θ).(zero(datay))
  xh[1] = 0.1
  yh[1] = 0.1
  for i in 2:length(datax)
      dxh = yh[i-1]
      dyh = θ * (1.0 - xh[i-1]^2) * yh[i-1] - xh[i-1]
      xh[i] ~ Normal(xh[i-1] + dxh * dt,ϕ*sqrt(dt))
      yh[i] ~ Normal(yh[i-1] + dyh * dt,ϕ*sqrt(dt))
  end
  datax ~ MvNormal(xh,σ)
  datay ~ MvNormal(yh,σ)
end

# ADVI
modelEMn = fitvpEMn(xn,yn)
Turing.setadbackend(:tracker)
advi = ADVI(10, 1000)
setchunksize(8)
q = vi(modelEMn, advi)

# sampling
z = rand(q, 500)
avg = vec(mean(z; dims = 2))

@test 0.5<avg[1]<1.5 # testing θ
@test 0.05<avg[2]<0.25 # tesing ϕ
