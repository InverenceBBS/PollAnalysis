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
using StateSpaceModels
using Plots
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

## Prepare data for analysis

Results = Polls[occursin.(r"election", Polls.Firm), :]
Results = @subset(Results, .!ismissing.(:date))
Results = @subset(Results, .!ismissing.(:value))


LocalResults = Results[occursin.(r"local", Results.Firm), :]

#### To be fixed: general election results appear twice
#### once as outcome of the current election table
#### once as outcome of the previous elections. 

NationalResults = Results[occursin.(r"general", Results.Firm), :]
EuropeanParliament = Results[occursin.(r"epe", Results.Firm), :]

Opinions = Polls[.!occursin.(r"election", Polls.Firm), :]
Opinions = @subset(Opinions, .!ismissing.(:date))
Opinions = @subset(Opinions, .!ismissing.(:value))


### Here we could use a weighting system to improve the averaging
Opinions = groupby(Opinions, [:date, :name]);
Opinions = @combine(Opinions, :value = mean(:value))

# Mono-party analysis

## Plot Trends

for this_party in unique(Opinions.name)
    @show this_party
    plt = plotResults(this_party, Opinions, NationalResults, LocalResults)
    savefig(plt, "./Plots/Trends/"*this_party*"_trends.png")
end

# Filter data with a local linear trend
# yₜ   = μₜ + γₜ + εₜ with εₜ ∼ N(0,σ²ₑ)
# μₜ₊₁ = μₜ + νₜ + ξₜ with ξₜ ∼ N(0,σₔ)
# νₜ₊₁ = νₜ + ζₜ      with ζₜ ∼ N(0,σ²ₛ)

K = 20

accuracies = DataFrame(
    ElectionDate = fill(Date,0),
    PrevisionDate = fill(Date,0),
    Party = fill(String,0),
    Result = fill(Float64,0),
    Prevision = fill(Float64,0),
    Error = fill(Float64,0)
)

this_party = "psoe"

for this_party in unique(Opinions.name)
    @show this_party
    this_party_Opinions, this_party_NationalResults, _ = extract_party_data(this_party,Opinions, NationalResults, LocalResults)   
    # all the magic happens here, where we apply a local linear trend state space model
    this_model = LocalLinearTrend(this_party_Opinions.value)
    fit!(this_model)
    this_filt = kalman_filter(this_model)

    this_filt_states = get_filtered_state(this_filt)[:,1]


    this_prev = forecast(this_model, 30)


    this_Plot = plot(this_model, this_prev, label=["observed" "prevision"])
    plot!(this_Plot, this_filt_states, label="Filtered")
   
    results_post_polls = @subset(this_party_NationalResults, :date .!= Date("2000-03-12"))

    
    if !(nrow(results_post_polls) == 0)
    
    rez_plot = plot(this_party_Opinions.date, this_filt_states, label="filtered")
    scatter!(rez_plot, this_party_NationalResults.date, this_party_NationalResults.value, color="green", label="National")
    savefig(rez_plot, "./Plots/LocalLinearFilter/"*this_party*"_filter_vs_rez.png")

    for election in eachrow(results_post_polls)
        
        k = minimum([count(this_party_Opinions.date .< election.date),K])

        if k > 0

        prevision_dates = find_previous_date(election.date, this_party_Opinions.date,k)
        
        Previsions = @subset DataFrame(date = this_party_Opinions.date, value = this_filt_states) @byrow begin
            :date ∈ prevision_dates
        end
    
        this_sse = [abs(prevision.value .- election.value) for prevision in eachrow(Previsions)]
    
        L = length(this_sse)  # Original length
            
        # Check that K is greater than L
        if k > L
            # Create a vector of zeros of the necessary length
            zero_padding = fill(missing,k - L)
            # Prepend the zeros to the original vector
            this_sse = vcat(zero_padding, this_sse)
        end

        this_accuracies = DataFrame(
            ElectionDate = fill(election.date,k),
            PrevisionDate = prevision_dates,
            Party = fill(this_party,k),
            Result = fill(election.value,k),
            Prevision = Previsions.value,
            Error = this_sse
            )

        global accuracies
        accuracies = vcat(accuracies,this_accuracies)
    end
    
    end

    end

    savefig(this_Plot, "./Plots/LocalLinearFilter/"*this_party*"_filter.png")
end


accuracies

CSV.write("./Data/Outputs/LocalLinearTrend_electoral_prediction_accuracy_by_party.csv",
accuracies
)


# df = accuracies[accuracies.Party .== "psoe",:]
# df = df[df.ElectionDate .== Date("2015-12-20"),:]
# scatter(df.ElectionDate .- df.PrevisionDate, df.Result .- df.Prevision)


# accuracies[1306,:]

# findmax((accuracies.Result .- accuracies.Prevision)./accuracies.Result)