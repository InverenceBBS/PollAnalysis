using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise

include("./functions.jl")


using Chain
using CSV
using DataFrames, DataFramesMeta
using Glob
using LinearAlgebra, Statistics
using Arrow
using Dates

# Read in data

Folder = "./Data/Raw"  
Files = glob("*.csv", Folder) 

Polls = DataFrame.(CSV.File.(Files,
        missingstring=["NA", "NAN", "NULL"],
        types=Dict(:value=>Float64)))

for i in 1:length(Polls)
    if "Lead" ∉ names(Polls[i])
        Polls[i][!,:Lead] .= missing
    end    
end


Polls = reduce(vcat,Polls)
Polls[!, :Lead] = Missings.coalesce.(string.(Polls[:, :Lead]), missing)

## Prepare data for analysis

Results = Polls[occursin.(r"election", Polls.Firm), :]
Results = @subset(Results, .!ismissing.(:date))
Results = @subset(Results, .!ismissing.(:value))


LocalResults = Results[occursin.(r"local", Results.Firm), :]
Arrow.write("./Data/Processed/LR.arr", LocalResults)


#### To be fixed: general election results appear twice
#### once as outcome of the current election table
#### once as outcome of the previous elections. 

NationalResults = Results[occursin.(r"general", Results.Firm), :]   
Arrow.write("./Data/Processed/NR.arr", NationalResults)


EuropeanParliament = Results[occursin.(r"epe", Results.Firm), :]
Arrow.write("./Data/Processed/EPE.arr", EuropeanParliament)


Opinions = Polls[.!occursin.(r"election", Polls.Firm), :]
Opinions = @subset(Opinions, .!ismissing.(:date))
Opinions = @subset(Opinions, .!ismissing.(:value))


### Here we could use a weighting system to improve the averaging
Opinions = groupby(Opinions, [:date, :name]);
Opinions = @combine(Opinions, :value = mean(:value))
Arrow.write("./Data/Processed/Opinions.arr", Opinions)