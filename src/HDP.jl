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


type CRFSample
    tji::Vector{Vector{Int}}
    njt::Vector{Vector{Int}}
    kjt::Vector{Vector{Int}}
    zz::Vector{Vector{Int}}

    CRFSample(tji::Vector{Vector{Int}}, njt::Vector{Vector{Int}}, kjt::Vector{Vector{Int}}, zz::Vector{Vector{Int}}) = new(tji, njt, kjt, zz)
end




function storesample{T}(
    hdp::HDP{T},
    KK_list::Vector{Int},
    KK_dict::Dict{Int, Vector{Vector{Int}}},
    alphas::Vector{Float64},
    betas::Vector{Vector{Float64}},
    gammas::Vector{Float64},
    n_burnins::Int, n_lags::Int, sample_n::Int,
    filename::ASCIIString)

    println("storing on disk...")
    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".jld")
    else
        dummy_filename = string(filename, "_", sample_n, ".jld")
    end

    JLD.save(dummy_filename,
        "hdp", hdp,
        "KK_list", KK_list,
        "KK_dict", KK_dict,
        "alphas", alphas, "betas", betas, "gammas", gammas,
        "n_burnins", n_burnins, "n_lags", n_lags, "sample_n", sample_n)
end


function storesample{T}(
    hdp::HDP{T},
    KK_list::Vector{Int},
    KK_dict::Dict{Int, CRFSample},
    n_burnins::Int, n_lags::Int, sample_n::Int,
    filename::ASCIIString)

    println("\nstoring on disk...\n")
    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".jld")
    else
        dummy_filename = string(filename, "_", sample_n, ".jld")
    end

    JLD.save(dummy_filename,
        "hdp", hdp,
        "KK_list", KK_list,
        "KK_dict", KK_dict,
        "n_burnins", n_burnins, "n_lags", n_lags, "sample_n", sample_n)
end


function sample_hyperparam!(hdp::HDP, n_group_j::Vector{Int}, m::Int)
    #= NOTE
    resampling the group level concentration parameter α0 using auxiliary variables
    w and s, Eq. 50 Teh etal 04 the Gamma distribution in Eq.50 is expressed using
    shape and rate. We have to rescale them to shape and scale representation to be
    able to use standard random gamma functions in Julia to draw from it. Also:
    Gamma(a, 1/b) = Gamma(a) / b
    =#

    n_groups = length(n_group_j)
    w = zeros(Float64, n_groups)
    for jj = 1:n_groups
        w[jj] = rand(Distributions.Beta(hdp.aa+1, n_group_j[jj]))
    end
    p = n_group_j / hdp.aa
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
function sample_hyperparam!(hdp::HDP, n_group_j::Vector{Int}, m::Int, n_internals::Int)
    for iteration = 1:n_internals
        sample_hyperparam!(hdp, n_group_j, m)
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
    KK_dict::Dict{Int, Vector{Vector{Int}}}=Dict{Int, Vector{Vector{Int}}}())


    # constructing components
    components = Array(typeof(hdp.component), hdp.KK)
    for kk = 1:hdp.KK
        components[kk] = deepcopy(hdp.component)
    end

    n_iterations    = n_burnins + (n_samples)*(n_lags+1)
    n_groups        = length(xx)
    n_group_j       = zeros(Int, n_groups)
    nn              = zeros(Int, n_groups, hdp.KK)
    pp              = zeros(Float64, hdp.KK+1)
    log_likelihood  = 0.0

    betas  = Array(Vector{Float64}, n_samples)
    gammas = zeros(Float64, n_samples)
    alphas = zeros(Float64, n_samples)

    for jj = 1:n_groups
        n_group_j[jj] = length(zz[jj])
    end

    if length(KK_list) == 0
        n_sample_old = 0
        KK_list = zeros(Int, n_samples)
        KK_dict = Dict{Int, Vector{Vector{Int}}}()
    else
        n_sample_old = length(KK_list)
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
                m = sum(M)
                sample_hyperparam!(hdp, n_group_j, m)
            end
        end # n_internals
        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = n_sample_old + convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[sample_n] = hdp.KK
            KK_dict[hdp.KK] = deepcopy(zz)
            alphas[sample_n] = hdp.aa
            betas[sample_n] = deepcopy(my_beta)
            gammas[sample_n] = hdp.gg
            if sample_n % store_every == 0
                storesample(hdp, KK_list, KK_dict, alphas, betas, gammas, n_burnins, n_lags, sample_n, filename)
            end
        end
    end # iteration
    KK_list, KK_dict, betas, gammas, alphas
end # collapsed_gibbs_sampler!


function CRF_gibbs_sampler!{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=100, filename::ASCIIString="HDP_results_",
    KK_list::Vector{Int}=Int[],
    KK_dict::Dict{Int, CRFSample}=Dict{Int, CRFSample}())



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
        KK_dict = Dict{Int, CRFSample}()
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
            end # tbl
        end # n_groups

        # resampling hyper-parameters
        if sample_hyperparam
            nn = zeros(Int, n_groups, hdp.KK)
            for jj = 1:n_groups
                for ii = 1:n_group_j[jj]
                    nn[jj, zz[jj][ii]] += 1
                end
            end
            m = sum([length(kjt[jj]) for jj=1:n_groups])
            sample_hyperparam!(hdp, n_group_j, m, n_internals)
        end




        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 &&  iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[sample_n] = hdp.KK
            crf_sample = CRFSample(deepcopy(tji), deepcopy(njt), deepcopy(kjt), deepcopy(zz))
            KK_dict[hdp.KK] = crf_sample
            if sample_n % store_every == 0
                storesample(hdp, KK_list, KK_dict, n_burnins, n_lags, sample_n, filename)
            end
        end


    end # iteration

    KK_list, KK_dict
end # CRF_gibbs_sampler



function posterior{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    KK_dict::Dict{Int, Vector{Vector{Int}}},
    KK::Int)

    n_groups  = length(xx)
    n_group_j = [length(xx[jj]) for jj = 1:n_groups]

    components = Array(typeof(hdp.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(hdp.component)
    end

    nn = zeros(Int, n_groups, KK)
    zz = KK_dict[KK]

    for jj = 1:n_groups
        for ii =     1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
        end
    end
    pij = nn + hdp.aa
    pij = pij ./ sum(pij, 1)


    [posterior(components[kk]) for kk =1:KK], nn, pij
end




function posterior{T1, T2}(
    hdp::HDP{T1},
    xx::Vector{Vector{T2}},
    KK_dict::Dict{Int, CRFSample},
    KK::Int,
    join_tables::Bool=false)


    n_groups  = length(xx)
    n_group_j = [length(xx[jj]) for jj = 1:n_groups]
    nn = zeros(Int, n_groups, KK)
    mm = zeros(Int, KK)

    components = Array(typeof(hdp.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(hdp.component)
    end

    tji = KK_dict[KK].tji
    njt = KK_dict[KK].njt
    kjt = KK_dict[KK].kjt
    zz  = KK_dict[KK].zz

    for jj = 1:n_groups
        for ii = 1:n_group_j[jj]
            kk = zz[jj][ii]
            additem!(components[kk], xx[jj][ii])
            nn[jj, kk] += 1
        end
    end

    pij = nn + hdp.aa
    pij = pij ./ sum(pij, 1)


    if join_tables
        for jj = 1:n_groups

            kk_list = unique(kjt[jj])

            if length(kk_list) != length(kjt[jj])
                new_kjt = zeros(Int, length(kk_list))
                new_njt = zeros(Int, length(kk_list))
                ll = 1

                for unq_kk in kk_list
                    idx = find(x -> x==unq_kk, kjt[jj])
                    new_kjt[ll] = unq_kk
                    new_njt[ll] = sum(njt[jj][idx])

                    for idid in idx
                        tbl_idx = find(x -> x==idid, tji[jj])
                        tji[jj][tbl_idx] = ll
                    end

                    ll += 1
                end

                kjt[jj] = new_kjt
                njt[jj] = new_njt
            end
        end
    end

    mm = zeros(Int, KK)
    for jj = 1:n_groups
        for kk in kjt[jj]
            mm[kk] += 1
        end
    end


    [posterior(components[kk]) for kk =1:KK], tji, njt, kjt, zz, nn, mm, pij

end
