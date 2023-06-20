function plotResults(party, Opinions, NationalResults, LocalResults)
    OpinionsParty, NationalResultsParty, LocalResultsParty = extract_party_data(party,Opinions, NationalResults, LocalResults)


    thisPlot = scatter(OpinionsParty.date, OpinionsParty.value, color="red", label="Polls", title = party)
    
    scatter!(thisPlot, NationalResultsParty.date, NationalResultsParty.value, color="blue", label="National")
    scatter!(thisPlot, LocalResultsParty.date, LocalResultsParty.value, color="green", label="Local")

    return(thisPlot)
end

function unify_results(party,ResDf)
    Res = @chain ResDf begin
        @subset(:name .== party)
        sort!(:date)
        @select(:date,:Firm,:name,:value)
        unique!
        end
    return(Res)
end

function extract_party_data(party,Opinions, NationalResults, LocalResults)
    OpinionsParty = @subset(Opinions, :name .== party)
    sort!(OpinionsParty, :date)

    NationalResultsParty = unify_results(party,NationalResults)
    
    LocalResultsParty = unify_results(party,LocalResults)

    return(OpinionsParty,NationalResultsParty,LocalResultsParty)
end


function kalman_fit(x0, ys;
                    P0=.5, Φ=1., b=.0, Q=.5, H=1., R=5.)
    
    # filter (assuming first observation at time 1)
    N = length(ys)
    
    p = Gaussian(x0, P0)
    ps = [p] # vector of filtered Gaussians
    for i in 1:N
        # global p
        # predict
        p = Φ*p ⊕ Gaussian(zero(x0), Q) #same as Gaussian(Φ*p.μ, Φ*p.Σ*Φ' + Q)
        # correct
        p, yres, _ = Kalman.correct(Kalman.JosephForm(), p, (Gaussian(ys[i], R), H))
        push!(ps, p) # save filtered density
    end

    return(ps,ys)

end
    
function plot_kf(ps,ys, party_name, dates)
    N = length(ys)
    p1 = scatter(dates, ys, color="red", label=party_name)
    plot!(p1, dates, [mean(p)[1] for p in ps[2:end]], color="blue", label="filtered x1", grid=false, ribbon=[var(p) for p in ps[2:end]], fillalpha=.5)

    return(p1)
end

function find_previous_date(target_date,set_dates, n=1)
    
    @assert n≥1

    @assert issorted(set_dates) "Dates are not sorted, this may cause problems"
    
    positives = set_dates[(target_date .- set_dates) .> zero(Day)]

    previous_date = n == 1 ? maximum(positives) : partialsort(positives, 1:n, rev = true)
    
    return(previous_date)
end