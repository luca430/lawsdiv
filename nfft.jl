#/ Start module
module Spectrum

using NFFT
using StatsBase
using Random

function Brownian(;
    n::Int = 512,
    return_positive=true,  #~ return positive domain of FFT
    nonuniform=true,       #~ use 'random' timepoints
    nonconsistent=false,   #~ make data 'non-consistent', by omitting some timepoints
    nmissing::Int = 64     #~ no. of missing datapoints when nonconsistent=true
)
    Random.seed!(n)
    t = range(0., 1., length=n)
    if nonuniform && !nonconsistent
        #/ Randomize timepoints a bit
        t = cumsum(abs.(randn(n)))
        tmin, tmax = extrema(t)
        t = @. (t - tmin) / (tmax - tmin)
    end
    
    #~ Brownian motion
    x = vcat(0.0, cumsum(randn(n - 1)) .* sqrt.(diff(t)))

    if nonconsistent && !nonuniform
        #/ Omit some timepoints
        idxs = sort(sample(1:n, n-nmissing, replace=false))
        t = t[idxs]
        x = x[idxs]
        n = length(t)
    end

    #~ Freq. nodes k
    #  as t∈[0,1], we can let Nf = 0.5 the Nyquist frequency
    Nf = 0.5
    k = range(-Nf, stop=Nf, length=n)

    #/ Compute non-uniform FFT
    p = NFFT.plan_nfft(k, n, reltol=1e-9)
    fhat = adjoint(p) * x
    
    #/ Compute (normalized) power spectrum
    S = abs2.(fhat) / n
    (!return_positive) && (return k, S)
    
    #/ Return only positive k (freqs)
    kc = findfirst(x -> x > 0, k)
    return k[kc:end], S[kc:end]
end

end # module Spectrum
#/ End module
