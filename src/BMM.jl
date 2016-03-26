#=
BMM.jl

10/27/2015
Adham Beyki, odinay@gmail.com
=#

##########################################
###       Bayesian Mixture Model       ###
##########################################
immutable BMM{T} <: MixtureModel
    component::T
    KK::Int
    aa::Float64

    BMM{T}(c::T, KK::Int, aa::Float64) = new(c, KK, aa)
end
BMM{T}(c::T, KK::Int, aa::Real) = BMM{typeof(c)}(c, KK, convert(Float64,    aa))


function Base.show(io::IO, bmm::BMM)
    println(io, "Finite Mixture Model with $(bmm.KK) $(typeof(bmm.component)) components")
end


function storesample{T}(
    bmm::BMM{T},
    components::Vector{T},
    zz::Vector{Int},
    n_burnins::Int, n_lags::Int, sample_n::Int,
    filename::ASCIIString)

    println("\nstoring on disk...\n")
    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".jld")
    else
        dummy_filename = string(filename, "_", sample_n, ".jld")
    end

    JLD.save(dummy_filename,
        "bmm", bmm,
        "components", components,
        "zz", zz,
        "n_burnins", n_burnins, "n_lags", n_lags, "sample_n", sample_n)
end


function collapsed_gibbs_sampler!{T1, T2}(
    bmm::BMM{T1},
    xx::Vector{T2},
    zz::Vector{Int},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    store_every::Int=100, filename::ASCIIString="BMM_results_")



    n_iterations   = n_burnins + (n_samples)*(n_lags+1)
    NN             = length(xx)
    nn             = zeros(Int, bmm.KK)
    pp             = zeros(Float64, bmm.KK)
    log_likelihood = 0.0

    # constructing the components
    components = Array(typeof(bmm.component), bmm.KK)
    for kk = 1:bmm.KK
        components[kk] = deepcopy(bmm.component)
    end


    # initializing the model
    tic()
    for ii = 1:NN
        kk = zz[ii]
        additem!(components[kk], xx[ii])
        nn[kk] += 1
        log_likelihood += loglikelihood(components[kk], xx[ii])
    end
    elapsed_time = toq()



    # starting the MCMC chain
    for iteration = 1:n_iterations

        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, time=%.2f, likelihood=%.2f",
            iteration, bmm.KK, elapsed_time, log_likelihood))
        log_likelihood = 0.0


        tic()
        for ii = randperm(NN)

            # 1
            # remove the datapoint
            kk = zz[ii]
            nn[kk] -= 1
            delitem!(components[kk], xx[ii])

            # 2
            # sample zz
            pp = zeros(Float64, bmm.KK)
            for kk = 1:bmm.KK
                pp[kk] = log(nn[kk] + bmm.aa) + logpredictive(components[kk], xx[ii])
            end
            lognormalize!(pp)
            kk = sample(pp)

            # 3
            # add the datapoint to the newly sampled cluster
            zz[ii] = kk
            nn[kk] += 1
            additem!(components[kk], xx[ii])
            log_likelihood += loglikelihood(components[kk], xx[ii])
        end # ii
        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))

            if sample_n % store_every == 0
                storesample(bmm, components, zz, n_burnins, n_lags, sample_n, filename)
            end
        end
    end # iteration

    sample_n = convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    storesample(bmm, components, zz, n_burnins, n_lags, sample_n, filename)
end



function posterior{T1, T2}(bmm::BMM{T1}, xx::Vector{T2}, zz::Vector{Int})

    components = Array(typeof(bmm.component), bmm.KK)
    for kk = 1:bmm.KK
        components[kk] = deepcopy(bmm.component)
    end

    NN = length(zz)
    nn = zeros(Int, bmm.KK)

    for ii = 1:NN
        kk = zz[ii]
        additem!(components[kk], xx[ii])
        nn[kk] += 1
    end

    return([posterior(components[kk]) for kk =1:bmm.KK], nn)
end
