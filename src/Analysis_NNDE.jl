using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise

using DataFrames, DataFramesMeta
using Dates

include("./functions.jl")
include("./99_read_in_data.jl")

using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using StaticArrays, ComponentArrays, Lux, Zygote, Plots, StableRNGs

rng = StableRNG(42)

# We propose first a bipartitic model of the form
# dXₜ   = NNₛ(Xₜ₋₁)dt
# where Xₜ  = [psoeₜ₋₁;ppₜ₋₁;...]

## We first prepare the data for eight parties
## we extract it from the processed data
## Join it all, matchin dates
parties = ["pp","psoe","vox","upd","ip","cs","up","sum","upo"]

just_party_data(party_name) = select.(extract_party_data(party_name,Opinions, NationalResults, LocalResults),:date,:value)

op_data, nr_data, lr_data = just_party_data(parties[1])

for party in parties[2:end]
    party_data_op, party_data_nr, party_data_lr = just_party_data(party)
    op_data = leftjoin(op_data,party_data_op,
                       on = :date, makeunique=true, renamecols = "" => party)
    nr_data = leftjoin(nr_data,party_data_nr,
                       on = :date, makeunique=true, renamecols = "" => party)
    lr_data = leftjoin(lr_data,party_data_lr,
                       on = :date, makeunique=true, renamecols = "" => party)
end

op_data = sort!(coalesce.(op_data, 0.),:date)
nr_data = sort!(coalesce.(nr_data, 0.),:date)
lr_data = sort!(coalesce.(lr_data, 0.),:date)


op_data.valueuposum = op_data.valuesum .+ op_data.valueupo
select!(op_data, Not([:valuesum, :valueupo]))

nr_data.valueuposum = nr_data.valuesum .+ nr_data.valueupo
select!(nr_data, Not([:valuesum, :valueupo]))

lr_data.valueuposum = lr_data.valuesum .+ lr_data.valueupo
select!(lr_data, Not([:valuesum, :valueupo]))

## We transform the data into matrices as we like them
## each column is a time step, each row a variable
Xₙ = Matrix(op_data[:,2:end])' |> collect
NR = Matrix(nr_data[:,2:end])' |> collect
U₀ = NR[:,1]

tspan = maximum(op_data.date) - minimum(op_data.date) |> Dates.value
t = (op_data.date |> sort |> unique) .- minimum(op_data.date) .|> Dates.value
t /= tspan
tspan = (0.0, 1.0)

rbf(x) = exp.(-(x .^ 2))



const NN = Lux.Chain(
        Lux.Dense(8, 12, rbf),
        Lux.Dense(12, 12, rbf),
        Lux.Dense(12, 13, rbf))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, NN)

function nn_dynamics!(du, u, p, t)
    û = NN(u, p, st)[1] # Network prediction
    du[1] = u[1]*û[1] - u[2]*û[9] - sum(u[3:end])*û[1]
    du[2] = u[2]*û[2] - u[1]*û[10] - sum(u[3:end])*û[1]
    du[3:end] .= u[3:end] .* û[3:8]
end

# Define the problem
prob_nn = ODEProblem(nn_dynamics!, U₀, tspan, p)

## prediction and loss functions
function predict(θ, X = U₀, T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol = 1e-6, reltol = 1e-6,
                sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂) #+ 0.8 * mean(abs2, NationalResults .- X̂) # Here's where we can add a penalisation based on National (and whatever else) results
end

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
     
        X̂ = predict(p, U₀, t)
        # Trained on noisy data vs observations
        plotto = scatter(t, transpose(Xₙ[1:2,:]), color = [:orange :lightblue ], label = ["Measurements" nothing], alpha = 0.2)
        plot!(plotto, t, transpose(X̂[1:2,:]), xlabel = "t", ylabel = "Sustain %", color = [:red :blue],
                         label = ["PSOE" "PP"])
        display(plotto)
    end
    return false
end


## TRAINING

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u

X̂ = predict(p_trained, U₀, t)
# Trained on noisy data vs real solution
scatter(t, transpose(Xₙ[[1,8],:]), color = [:orange :lightblue ], label = ["Measurements" nothing], alpha = 0.2)
plot!(t, transpose(X̂[7:8,:]), xlabel = "t", ylabel = "Sustain %", color = [:red :blue],
                     label = ["PP" "PSOE"])