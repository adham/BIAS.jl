#=
LDA.jl

Adham Beyki, odinay@gmail.com
27/10/2015
=#

##########################################
###### Latent Dirichlet Allocation ######
##########################################
immutable LDA{T} <: hMixtureModel
    component::T
    KK::Int
    aa::Float64

    LDA{T}(c::T, KK::Int, aa::Float64) = new(c, KK, aa)
end
LDA{T}(c::T, KK::Int, aa::Real) = LDA{typeof(c)}(c, KK, convert(Float64,    aa))


function Base.show(io::IO, lda::LDA)
    println(io, "Latent Dirichlet Allocation model with $(lda.KK) $(typeof(lda.component)) components")
end


function LDA_sample_zz{T1, T2}(
    components::Vector{T1},
    xx::T2,
    nn::Matrix{Int},
    pp::Vector{Float64},
    aa::Float64,
    jj::Int)

  KK = length(pp)
  @inbounds for kk = 1:KK
    pp[kk] = log(nn[jj, kk] + aa) + logpredictive(components[kk], xx)
  end

  lognormalize!(pp)
    kk = sample(pp)
  return kk
end


function storesample{T}(
        lda::LDA{T},
        components::Vector{T},
        zz::Vector{Vector{Int}},
        i::Int,
        iteration::Int,
        filename::ASCIIString)

    println("storing on disk...")
    if endswith(filename, "_")
        dummy_filename = string(filename, i, ".jld")
    else
        dummy_filename = string(filename, "_", i, ".jld")
    end

    JLD.save(dummy_filename,
        "n_smaple", i,
        "lda", lda,
        "components", components,
        "zz", zz,
        "iteration", iteration)
end


function collapsed_gibbs_sampler!{T1, T2}(
    lda::LDA{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
        store_every::Int=100, filename::ASCIIString="LDA_results_")


    components = Array(typeof(lda.component), lda.KK)
    for kk = 1:lda.KK
        components[kk] = deepcopy(lda.component)
    end

    n_groups       = length(xx)
  n_group_j      = [length(zz[jj]) for jj = 1:n_groups]
  n_iterations   = n_burnins + (n_samples)*(n_lags+1)
  lda_aa         = fill(lda.aa, lda.KK)
  pp             = zeros(Float64, lda.KK)
  nn             = zeros(Int, n_groups, lda.KK)
    log_likelihood = 0.0


  # initializing the components
    tic()
  for jj = 1:n_groups
    for ii = 1:n_group_j[jj]
      kk = zz[jj][ii]
      additem!(components[kk], xx[jj][ii])
      nn[jj, kk] += 1
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])
    end
  end
  elapsed_time = toq()


  # starting the MCMC chain
    for iteration = 1:n_iterations

        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, time=%.2f, likelihood=%.2f",
            iteration, lda.KK, elapsed_time, log_likelihood))
    log_likelihood = 0.0

        tic()
    @inbounds for jj = randperm(n_groups)
      @inbounds for ii = randperm(n_group_j[jj])

                # 1
                # remove the datapoint
                kk = zz[jj][ii]
        delitem!(components[kk], xx[jj][ii])
        nn[jj, kk] -= 1

        # 2
                # sample zz
                kk = LDA_sample_zz(components, xx[jj][ii], nn, pp, lda.aa, jj)

                # 3
                # add the datapoint to the newly sampled cluster
        zz[jj][ii] = kk
        additem!(components[kk], xx[jj][ii])
        nn[jj, kk] += 1
                log_likelihood += loglikelihood(components[kk], xx[jj][ii])
      end
    end
    elapsed_time = toq()

        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            i = convert(Int, (iteration-n_burnins)/(n_lags+1))
            # sample should be saved here

            if i % store_every == 0
                storesample(lda, components, zz, i, iteration, filename)
            end
        end
  end
end


function posterior{T1, T2}(lda::LDA{T1},
        xx::Vector{Vector{T2}}, zz::Vector{Vector{Int}})

  n_groups = length(xx)
  n_group_j = [length(zz[jj]) for jj = 1:n_groups]
  components = [deepcopy(lda.component) for k = 1:lda.KK]
  nn = zeros(Int, n_groups, lda.KK)

  for jj = 1:n_groups
    for ii = 1:n_group_j[jj]
      kk = zz[jj][ii]
      additem!(components[kk], xx[jj][ii])
      nn[jj, kk] += 1
    end
  end

  return([posterior(components[kk]) for kk =1:lda.KK], nn)
end




