module DataImport

using RData
using DataFrames, DataFramesMeta, GLM
using Statistics, StatsBase

function GetLongData(path; min_samples=1, min_counts=1, min_nreads=1)
    raw_data = load(path)
    proj_time = raw_data["proj_time"]

    # Fix environments with inverted nreads/count
    mask = proj_time.nreads .< proj_time.count
    for i in eachindex(mask)
        if mask[i]
            tmp = proj_time.nreads[i]
            proj_time.nreads[i] = proj_time.count[i]
            proj_time.count[i] = tmp
        end
    end

    # Filter by min_samples
    grouped = groupby(proj_time, :project_id)
    filtered_data = combine(grouped) do sdf
        if length(unique(sdf.sample_id)) > min_samples
            return sdf
        else
            return DataFrame()
        end
    end
    
    # Filter by min_nreads
    filtered_data = proj_time[proj_time.nreads .> min_nreads, :]
    filtered_data = proj_time[proj_time.count .> min_counts, :]

    # Drop unnecessary columns
    select!(filtered_data, Not([:host_age, :date_collection]))

    # Separate by host
    M3 = filtered_data[filtered_data.host_id .== "M3", :]
    F4 = filtered_data[filtered_data.host_id .== "F4", :]
    hosts = Dict("M3" => M3, "F4" => F4)

    # Separate by environment
    sep_data = Dict()
    for key in keys(hosts)
        df = hosts[key]
        FECES = df[df.classification .== "feces", :]
        ORALCAVITY = df[df.classification .== "oralcavity", :]
        L_PALM = df[(df.classification .== "skin") .& (df.samplesite .== "L_palm"), :]
        R_PALM = df[(df.classification .== "skin") .& (df.samplesite .== "R_palm"), :]
        
        select!(FECES, Not([:host_id, :classification, :samplesite]))
        sort!(FECES, [:experiment_day, :otu_id])
        select!(ORALCAVITY, Not([:host_id, :classification, :samplesite]))
        sort!(ORALCAVITY, [:experiment_day, :otu_id])
        select!(L_PALM, Not([:host_id, :classification, :samplesite]))
        sort!(L_PALM, [:experiment_day, :otu_id])
        select!(R_PALM, Not([:host_id, :classification, :samplesite]))
        sort!(R_PALM, [:experiment_day, :otu_id])
        
        sep_data[key] = Dict("FECES" => FECES, "ORALCAVITY" => ORALCAVITY, "L_PALM" => L_PALM, "R_PALM" => R_PALM)
    end

    return sep_data
end

function ComputeLongFreqs(df)
    # Compute relative abundance per run
    df.rel_abundance = df.count ./ df.nreads

    # Compute tf (mean relative abundance when present) and o (occupancy)
    agg = combine(groupby(df, [:otu_id, :experiment_day])) do sdf
        # All relative abundances (some may be 0)
        abundances = sdf.rel_abundance

        # Detected runs (non-zero counts)
        detected = sdf.count .> 0
        tf = mean(abundances[detected])           # mean abundance when detected
        o = sum(detected) / length(abundances)    # fraction of runs with detection

        (; otu_id = sdf.otu_id[1],
           experiment_day = sdf.experiment_day[1],
           tf = tf,
           o = o,
           f = tf * o)
    end

    return agg[:, [:otu_id, :experiment_day, :f]]
end


end # end module
















