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

end # end module
















