#=
HDP.jl

Adham Beyki, odinay@gmail.com
27/10/2015

TODO:
implement save_samples for CRF
implement resample_hyperparams for CRF
=#

###################################################
## Hierarchical Dirichlet Process Mixture Models ##
###################################################
type HDP{T} <: hMixtureModel
    component::T
    KK::Int
    gg::Float64
    g1::Float64
    g2::Float64
    aa::Float64
    a1::Float64
    a2::Float64

    HDP{T}(c::T,
           KK::Int,
           gg::Float64, g1::Float64, g2::Float64,
           aa::Float64, a1::Float64, a2::Float64) = new(c, KK, gg, g1, g2, aa, a1, a2)
end

HDP{T}(c::T, KK::Int, gg::Real, g1::Real, g2::Real, aa::Real, a1::Real, a2::Real) = HDP{typeof(c)}(c, KK,
    convert(Float64, gg), convert(Float64, g1), convert(Float64, g2),
    convert(Float64, aa), convert(Float64, a1), convert(Float64, a2))


function Base.show(io::IO, hdp::HDP)
  println(io, "Hierarchical Dirichlet Process Mixture Model with $(hdp.KK) $(typeof(hdp.component)) components")
end


function storesample{T}(
    hdp::HDP{T},
    KK_list::Vector{Int},
    KK_zz_dict::Dict{Int, Vector{Vector{Int}}},
    alphas::Vector{Float64},
    betas::Vector{Vector{Float64}},
    gammas::Vector{Float64},
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
        "hdp", hdp,
        "KK_list", KK_list,
        "KK_zz_dict", KK_zz_dict,
        "alphas", alphas, "betas", betas, "gammas", gammas,
        "iteration", iteration)
end # storesample


function sample_hyperparam!(hdp::HDP, nn_sum::Vector{Int}, m::Int)
    #= NOTE
    resampling the group level concentration parameter α0 using auxiliary variables
    w and s, Eq. 50 Teh etal 04 the Gamma distribution in Eq.50 is expressed using
    shape and rate. We have to rescale them to shape and scale representation to be
    able to use standard random gamma functions in Julia to draw from it. Also:
    Gamma(a, 1/b) = Gamma(a) / b
    =#

    n_groups = length(nn_sum)
    w = zeros(Float64, n_groups)
    for jj = 1:n_groups
        w[jj] = rand(Distributions.Beta(hdp.aa+1, nn_sum[jj]))
    end
    p = nn_sum / hdp.aa
    p ./= (p+1.0)

    s = zeros(Int, n_groups)
    for jj = 1:n_groups
        s[jj] = rand(Distributions.Binomial(1, p[jj]))
    end

    aa_shape = hdp.a1 + m - sum(s)
    aa_rate  = hdp.a2 - sum(log(w))
    hdp.aa = rand(Distributions.Gamma(aa_shape)) / aa_rate

    # resampling the top level concentration parameter γ, Escobar and West 95
    eta = rand(Distributions.Beta(hdp.gg+1, m))
    pi_eta = 1 / (1 + (m*(hdp.g2 - log(eta))) / (hdp.g1 + hdp.KK - 1))

    if rand() < pi_eta
        hdp.gg = rand(Distributions.Gamma(hdp.g1+hdp.KK)) / (hdp.g2-log(eta))
    else
        hdp.gg = rand(Distributions.Gamma(hdp.g1+hdp.KK-1)) / (hdp.g2-log(eta))
    end
end # sample_hyperparam


function collapsed_gibbs_sampler!{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=100, filename::ASCIIString="HDP_results_",
    KK_list::Vector{Int}=Int[],
    KK_zz_dict::Dict{Int, Vector{Vector{Int}}}=Dict{Int, Vector{Vector{Int}}}())


    # constructing components
    components = Array(typeof(hdp.component), hdp.KK)
    for kk = 1:hdp.KK
        components[kk] = deepcopy(hdp.component)
    end

    n_iterations    = n_burnins + (n_samples)*(n_lags+1)
    n_groups        = length(xx)
    n_group_j       = [length(zz[jj]) for jj = 1:n_groups]
    nn              = zeros(Int, n_groups, hdp.KK)
    pp              = zeros(Float64, hdp.KK+1)
    log_likelihood  = 0.0

    betas  = Array(Vector{Float64}, n_samples)
    gammas = zeros(Float64, n_samples)
    alphas = zeros(Float64, n_samples)


    if length(KK_list) == 0
        KK_list = zeros(Int, n_samples)
        KK_zz_dict = Dict{Int, Vector{Vector{Int}}}()
    else
        KK_list = vcat(KK_list, zeros(Int, n_samples))
    end

    snumbers_file = string(Pkg.dir(), "\\BIAS\\src\\StirlingNums_10K.mat")
    snumbers_data = MAT.matread(snumbers_file)
    snumbers = snumbers_data["snumbersNormalizedSparse"]


    ################################
    #    Initializing the model    #
    ################################
    tic()
    print_with_color(:red, "\nInitializing the model\n")
    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])
        end
    end
    KK_list[1] = hdp.KK
    elapsed_time = toq()

    my_beta = ones(Float64, hdp.KK+1) / (hdp.KK+1)



    ###################################
    #     starting the MCMC chain     #
    ###################################
    for iteration = 1:n_iterations

        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, gg=%.2f, aa=%.2f, time=%.2f, likelihood=%.2f",
            iteration, hdp.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]),
            hdp.gg, hdp.aa, elapsed_time, log_likelihood))
        log_likelihood = 0.0

        tic()

        # make sure all the components are active
        for kk = 1:hdp.KK
            if sum(nn[:, kk]) == 0
                println("\tcomponent $kk has become inactive")
                nn = del_column(nn, kk)
                splice!(components, kk)

                for ll in 1:n_groups
                    idx = find(x -> x>kk, zz[ll])
                    zz[ll][idx] -= 1
                end

                my_beta[hdp.KK+1] += my_beta[kk]
                splice!(my_beta, kk)
                splice!(pp, kk)
                hdp.KK -= 1
            end
        end

        #=
        Now
        1. resample zz[jj][ii]
            for all jj, ii
                1. remove xx[jj][ii] from its cluster
                2. resample zz[jj][ii]
                3. add x[jj][ii] to the resampled cluster
            end
        2. resample β
        3. resample hyperparams γ, α0
        =#

        # sample zz[jj][ii]
        for jj = randperm(n_groups)
            for ii = randperm(n_group_j[jj])


                ##   1.1   ##
                #  removing the observation
                kk = zz[jj][ii]
                delitem!(components[kk], xx[jj][ii])
                nn[jj, kk] -= 1

                # If as the result of removing xx[jj][ii]
                # θ_k becomes inactive, remove the atom
                if sum(nn[:, kk]) == 0
                    println("\tcomponent $kk has become inactive")
                    nn = del_column(nn, kk)
                    splice!(components, kk)

                    # shifting zz > kk back
                    for ll in 1:n_groups
                        idx = find(x -> x>kk, zz[ll])
                        zz[ll][idx] -= 1
                    end

                    # removing the stick kk
                    my_beta[hdp.KK+1] += my_beta[kk]
                    splice!(my_beta, kk)
                    splice!(pp, kk)
                    hdp.KK -= 1
                end


                ##   1.2   ##
                # resample kk
                for ll = 1:hdp.KK
                    pp[ll] = log(nn[jj, ll] + hdp.aa * my_beta[ll]) + logpredictive(components[ll], xx[jj][ii])
                end
                pp[hdp.KK+1] = log(hdp.aa * my_beta[hdp.KK+1]) + logpredictive(hdp.component, xx[jj][ii])
                lognormalize!(pp)
                kk = sample(pp)

                # if kk is new, we instantiate a new cluster
                if kk == hdp.KK+1
                    println("\tcomponent $(kk) activated.")
                    push!(components, deepcopy(hdp.component))
                    nn = add_column(nn)

                    b = rand(Distributions.Beta(1, hdp.gg))
                    b_new = my_beta[hdp.KK+1]
                    my_beta[hdp.KK+1] = b * b_new
                    push!(my_beta, (1-b)*b_new)
                    push!(pp, 0.0)
                    hdp.KK += 1
                end

                ##   1.3   ##
                # adding the data point to the resampled cluster
                zz[jj][ii] = kk
                additem!(components[kk], xx[jj][ii])
                nn[jj, kk] += 1
                log_likelihood += loglikelihood(components[kk], xx[jj][ii])

            end # n_group_j
        end # n_groups

        M = zeros(Int, n_groups, hdp.KK)
        for hh in 1:n_internals
        ##   2  ##
        # resampling β vector using auxiliary variable method
        # Eq. 40, 41 Teh etal 2004
            for jj = 1:n_groups
                for kk = 1:hdp.KK
                    if nn[jj, kk] == 0
                        M[jj, kk] = 0
                    else
                        rr = zeros(Float64, nn[jj, kk])
                        for mm = 1:nn[jj, kk]
                            rr[mm] = log(snumbers[nn[jj, kk], mm]) + mm*log(hdp.aa * my_beta[kk])
                        end
                        lognormalize!(rr)
                        M[jj, kk] = sample(rr)
                    end
                end # kk
            end # n_groups
            MM = convert(Vector{Float64}, vec(sum(M, 1)))
            push!(MM, hdp.gg)
            my_beta = rand(Distributions.Dirichlet(MM))


            ##   3   ##
            # resampling hyper-parameters γ and α0
            if sample_hyperparam
                nn_sum = vec(sum(nn, 2))
                m = sum(M)
                sample_hyperparam!(hdp, nn_sum, m)
            end
        end # n_internals
        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            i = convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[i] = hdp.KK
            KK_zz_dict[hdp.KK] = deepcopy(zz)
            betas[i] = deepcopy(my_beta)
            gammas[i] = hdp.gg
            alphas[i] = hdp.aa
            if i % store_every == 0
                storesample(hdp, KK_list, KK_zz_dict, alphas, betas, gammas, i, iteration, filename)
            end
        end
    end # iteration
    KK_list, KK_zz_dict, betas, gammas, alphas
end # collapsed_gibbs_sampler!


function CRF_gibbs_sampler!{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=100, filename::ASCIIString="HDP_results_",
    KK_list::Vector{Int}=Int[],
    KK_zz_dict::Dict{Int, Vector{Vector{Int}}}=Dict{Int, Vector{Vector{Int}}}())



    # constructing the components
    components = Array(typeof(hdp.component), hdp.KK)
    for kk = 1:hdp.KK
        components[kk] = deepcopy(hdp.component)
    end

    n_iterations = n_burnins + (n_samples)*(n_lags+1)
    n_groups     = length(xx)
    n_group_j    = [length(zz[jj]) for jj = 1:n_groups]
    tji          = Array(Vector{Int}, n_groups)     # tji[jj][ii] represents the table that customer ii in restaurant jj sits at
    njt          = Array(Vector{Int}, n_groups)     # njt[jj][tbl] represents the number of customers sitting at table tbl in restaurant jj
    kjt          = Array(Vector{Int}, n_groups)     # kjt[jj][tbl] represents the dish being served at table tbl at restaurant jj
    mm           = zeros(Int, hdp.KK)                # mm[kk] represents the number of times dish kk is ordered



    if length(KK_list) == 0
        KK_list = zeros(Int, n_samples)
        KK_zz_dict = Dict{Int, Vector{Vector{Int}}}()
    else
        KK_list = vcat(KK_list, zeros(Int, n_samples))
    end




    ##############################
    #   Initializing the model   #
    ##############################
    tic()

    print_with_color(:red, "\nInitializing the model\n")
    log_likelihood = 0.0

    for jj = 1:n_groups

        kjt[jj] = unique(zz[jj])        # table settings in group jj
        n_tbls  = length(kjt[jj])       # number of tables in group jj

        tji[jj] = zeros(Int, n_group_j[jj])
        njt[jj] = zeros(Int, n_tbls)

        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            log_likelihood += loglikelihood(components[kk], xx[jj][ii])

            tbl = findfirst(kjt[jj], kk)
            tji[jj][ii] = tbl
            njt[jj][tbl] += 1
        end
    end

    for jj = 1:n_groups
        for kk in kjt[jj]
            mm[kk] += 1
        end
    end
    KK_list[1] = hdp.KK
    elapsed_time = toq()




    ###################################
    #     starting the MCMC chain     #
    ###################################
    for iteration = 1:n_iterations

        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end

        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, gg=%.2f, aa=%.2f, time=%.2f, likelihood=%.2f",
            iteration, hdp.KK, indmax(hist(KK_list, .5:maximum(KK_list)+0.5)[2]),
            hdp.gg, hdp.aa, elapsed_time, log_likelihood))

        log_likelihood = 0.0



        ######################################
        # 1. resampling tables for customers #
        ######################################
        for jj = 1:n_groups

            n_tbls = length(kjt[jj])

            for ii = 1:n_group_j[jj]

                # remove the effect of the data point
                kk = zz[jj][ii]
                tbl = tji[jj][ii]
                delitem!(components[kk], xx[jj][ii])
                njt[jj][tbl] -= 1

                # if by removing xx[jj][ii] table tji[jj][ii] becomes empty
                # remove the table as well
                if njt[jj][tbl] == 0

                    n_tbls -= 1
                    mm[kk] -= 1

                    splice!(kjt[jj], tbl)
                    splice!(njt[jj], tbl)

                    idx = find(x -> x>tbl, tji[jj])
                    tji[jj][idx] -= 1

                    # if tji[jj][ii] (which we have removed it) is the only table which is
                    # serving dish kk and by removing it no other table serves dish kk,
                    # remove dish kk from menu
                    if mm[kk] == 0

                        hdp.KK -= 1
                        splice!(mm, kk)
                        splice!(components, kk)

                        for ll = 1:n_groups
                            idx = find(x -> x>kk, zz[ll])
                            zz[ll][idx] -= 1

                            idx = find(x -> x>kk, kjt[ll])
                            kjt[ll][idx] -= 1
                        end
                    end # if mm[kk]
                end # if njt

                # resample table tji[jj][ii]
                pp = zeros(Float64, n_tbls+1)
                for tbl = 1:n_tbls
                    kk = kjt[jj][tbl]
                    pp[tbl] = log(njt[jj][tbl]) + logpredictive(components[kk], xx[jj][ii])
                end
                pp[n_tbls+1] = log(hdp.aa) + logpredictive(hdp.component, xx[jj][ii])

                lognormalize!(pp)
                tbl = sample(pp)

                # if a new table is selected, draw the dish for the new table
                if tbl == n_tbls+1

                    n_tbls += 1
                    push!(kjt[jj], 0)
                    push!(njt[jj], 0)

                    pp = zeros(Float64, hdp.KK+1)
                    for kk = 1:hdp.KK
                        pp[kk] = log(mm[kk]) + logpredictive(components[kk], xx[jj][ii])
                    end
                    pp[hdp.KK+1] = log(hdp.gg) + logpredictive(hdp.component, xx[jj][ii])

                    lognormalize!(pp)
                    kk = sample(pp)

                    if kk == hdp.KK+1
                        hdp.KK += 1
                        push!(components, deepcopy(hdp.component))
                        push!(mm, 0)
                    end

                    kjt[jj][tbl] = kk
                    mm[kk] += 1
                end

                kk = kjt[jj][tbl]
                tji[jj][ii] = tbl
                njt[jj][tbl] += 1
                additem!(components[kk], xx[jj][ii])
                zz[jj][ii] = kk
            end # n_group_j[jj]
        end # n_groups


        ########################
        # 2. resampling dishes #
        ########################
        for jj = 1:n_groups

            # number of tables in restaurant jj
            n_tbls = length(kjt[jj])

            for tbl = 1:n_tbls

                kk = kjt[jj][tbl]
                mm[kk] -= 1

                tidx = find(x -> x==tbl, tji[jj])

                # if tbl is the only table serving dish kk, by removing it we need to
                # remove the dish from the menu

                if mm[kk] == 0
                    hdp.KK -= 1
                    splice!(mm, kk)
                    splice!(components, kk)

                    for ll = 1:n_groups
                        idx = find(x -> x>kk, zz[ll])
                        zz[ll][idx] -= 1

                        idx = find(x -> x>kk, kjt[ll])
                        kjt[ll][idx] -= 1
                   end

               else
                    # otherwise we need to remove the customers who were sitting at table tbl in
                    # jj, having dish kk, one by one
                    for ll in tidx
                       delitem!(components[kk], xx[jj][ll])
                    end
               end


                # resampling the dish
                pp = zeros(Float64, hdp.KK+1)
                for kk = 1:hdp.KK
                    pp[kk] = log(mm[kk])
                    for ll in tidx
                        pp[kk] += logpredictive(components[kk], xx[jj][ll])
                    end
                end
                pp[hdp.KK+1] += log(hdp.gg)
                for ll in tidx
                    pp[hdp.KK+1] += logpredictive(hdp.component, xx[jj][ll])
                end

                lognormalize!(pp)
                kk = sample(pp)

                if kk == hdp.KK+1
                    hdp.KK += 1
                    push!(components, deepcopy(hdp.component))
                    push!(mm, 0)
                end

                kjt[jj][tbl] = kk
                for ll in tidx
                    additem!(components[kk], xx[jj][ll])
                end
                mm[kk] += 1
                zz[jj][tidx] = kk
            end # n_group_j[jj]
        end # n_groups


        # TODO
        # resample hyperparams

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            i = convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[i] = hdp.KK
            KK_zz_dict[hdp.KK] = deepcopy(zz)
            if i % store_every == 0
                # TODO
                # save function
            end
        end


    end # iteration

    KK_list, KK_zz_dict
end # CRF_gibbs_sampler


function posterior{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    KK_zz_dict::Dict{Int, Vector{Vector{Int}}},
    K::Int)

    n_groups  = length(xx)
    n_group_j = [length(xx[jj]) for jj = 1:n_groups]

    components = Array(typeof(hdp.component), K)
    for kk = 1:K
        components[kk] = deepcopy(hdp.component)
    end

    nn = zeros(Int, n_groups, K)
    zz = KK_zz_dict[K]

    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
        end
    end
    pij = nn + hdp.aa
    pij = pij ./ sum(pij, 1)


    return([posterior(components[kk]) for kk =1:K], nn, pij)
end
