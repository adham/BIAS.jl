
#=
RCRP.jl

11/26/2015
Adham Beyki, odinay@gmail.com
=#


###################################################
###### Recurrent Chinese Restaurant Process  ######
###################################################

type RCRP{T} <: MixtureModel
    component::T
    KK::Int
    aa::Vector{Float64}
    a1::Float64
    a2::Float64

    RCRP{T}(c::T, KK::Int, aa::Vector{Float64}, a1::Float64, a2::Float64) = new(c, KK, aa, a1, a2)
end
RCRP{T}(c::T, KK::Int, aa::Real, TT::Int, a1::Real, a2::Real) = RCRP{typeof(c)}(c, KK, fill(convert(Float64, aa), TT),
    convert(Float64, a1), convert(Float64, a2))

function Base.show(io::IO, rcrp::RCRP)
    println(io, "Recurrent Chinese Restaurant Process with $(rcrp.KK) $(typeof(rcrp.component)) components")
end

function storesample{T}(
        rcrp::RCRP{T},
        KK_list::Vector{Int},
        KK_zz_dict::Dict{Int, Vector{Vector{Int}}},
        n_burnins::Int, n_lags::Int, sample_n::Int,
        filename::ASCIIString)

    
    println("\nstoring on disk...\n")
    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".jld")
    else
        dummy_filename = string(filename, "_", sample_n, ".jld")
    end

    JLD.save(dummy_filename,
        "rcrp", rcrp,
        "KK_list", KK_list,
        "KK_zz_dict", KK_zz_dict,
        "n_burnins", n_burnins, "n_lags", n_lags, "sample_n", sample_n)
end



function evolve{T}(phi::T)
    ## given phi[t, k] it returns phi[t+1, k]
    ## p(phi[t+1, k] | phi[t, k])

    evolved_phi = deepcopy(phi)
    evolved_phi
end


function sample_hyperparam!(rcrp::RCRP, tt::Int, n::Int, KK::Int, iters::Int)

    @inbounds for n = 1:iters
        eta = rand(Distributions.Beta(rcrp.aa[tt]+1, n))

        rr = (rcrp.a1+KK-1) / (n*(rcrp.a2-log(eta)))
        pi_eta = rr / (1.0+rr)

        if rand() < pi_eta
            rcrp.aa[tt] = rand(Distributions.Gamma(rcrp.a1+KK)) / (rcrp.a2-log(eta))
        else
            rcrp.aa[tt] = rand(Distributions.Gamma(rcrp.a1+KK-1)) / (rcrp.a2-log(eta))
        end
    end
end


function RCRP_gibbs_sampler!{T1, T2}(
    rcrp::RCRP{T1},
    xx::Vector{Vector{T2}},
    zz::Vector{Vector{Int}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true,  n_internals::Int=10,
    store_every::Int=100, filename::ASCIIString="RCRP_results_",
    KK_list::Vector{Int}=Int[],
    KK_zz_dict::Dict{Int, Vector{Vector{Int}}}=Dict{Int, Vector{Vector{Int}}}())

    # number of epochs
    TT = length(xx)

    # number of observations in epoch tt
    N_t = Array(Int, TT)
    for tt = 1:TT
        N_t[tt] = length(xx[tt])
    end



    if length(KK_list) == 0
        KK_list = zeros(Int, n_samples)
        KK_zz_dict = Dict{Int, Vector{Vector{Int}}}()
    else
        KK_list = vcat(KK_list, zeros(Int, n_samples))
    end



    # construct the observation count
    nn = zeros(Int, TT, rcrp.KK)
    for tt = 1:TT
        @inbounds for ii = 1:N_t[tt]
            nn[tt, zz[tt][ii]] += 1
        end
    end


    # make sure there is no inactive chain throughout all epochs
    kk = 1
    while kk <= rcrp.KK
        if sum(nn[:, kk]) == 0
            println("\tcomponent $kk has become inactive")
            nn = del_column(nn, kk)

            for tt = 1:TT
                idx = find(x -> x>kk, zz[tt])
                zz[tt][idx] -= 1
            end

            rcrp.KK -= 1
        end
        kk += 1
    end


    # construct the table indices
    # zz2table[tt, kk] returns the table index of chain kk at epoch tt
    zz2table = zeros(Int, TT, rcrp.KK)
    k_t = 1
    for kk = 1:rcrp.KK
        for tt = 1:TT
            if length(find(x -> x==kk, zz[tt])) != 0
                zz2table[tt, kk] = maximum(zz2table[tt, :]) + 1
            end
        end
    end



    ###############################
    # constructing the components #
    ###############################
    # parameters (dishes) of the first epoch are drawn from G_0, but @inbounds for
    # epochs t in [2, KK], the parameters are either an evolved from t-1
    # or drawn from G_0
    components = Array(Vector{typeof(rcrp.component)}, TT)

    for tt = 1:TT
        K_t = maximum(zz2table[tt, :])
        components[tt] = Array(typeof(rcrp.component), K_t)
        for k_t = 1:K_t
            components[tt][k_t] = deepcopy(rcrp.component)
        end
    end

    # Now add the observations
    log_likelihood = 0.0
    tic()
    for tt = 1:TT
        for ii = 1:N_t[tt]
            k_t = zz2table[tt, zz[tt][ii]]
            additem!(components[tt][k_t], xx[tt][ii])
            log_likelihood += loglikelihood(components[tt][k_t], xx[tt][ii])
        end
    end
    KK_list[1] = rcrp.KK
    elapsed_time = toq()




    ############################
    ##          MCMC          ##
    ############################
    n_iterations = n_burnins + (n_samples)*(n_lags+1)

    @inbounds for iteration = 1:n_iterations

        # verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, aa=%.2f, time=%.2f, likelihood=%.2f", iteration, rcrp.KK,
            indmax(hist(KK_list, 0.:maximum(KK_list)+0.5)[2]), rcrp.aa[1], elapsed_time, log_likelihood))



        #####################
        ##     epoch 1     ##
        #####################
        log_likelihood = 0.0
        tic()
        tt = 1

        ## L2 is used in computing p(z_{t+1} | z{t})
        L2 = sum(log(collect(0 : N_t[tt+1]-1) + N_t[tt] + rcrp.aa[tt]))


        # iterating over observations
        @inbounds for ii = 1:N_t[tt]


            kk  = zz[tt][ii]                # cluster id
            k_t = zz2table[tt, kk]          # table id

            # remove the observation
            delitem!(components[tt][k_t], xx[tt][ii])
            nn[tt, kk] -= 1

            if nn[tt, kk] == 0
                zz2table[tt, kk] = 0
                splice!(components[tt], k_t)

                idx = find(x -> x>k_t, zz2table[tt, :])
                zz2table[tt, idx] -= 1

                if sum(nn[:, kk]) == 0
                    @inbounds for tt_ = 1:TT

                        k_t = zz2table[tt_, kk]
                        if k_t != 0
                            splice!(components[tt_], k_t)
                            idx = find(x -> x>k_t, zz2table[tt, :])
                            zz2table[tt_, idx] -= 1
                            zz2table[tt_, kk] = 0
                        end

                        idx = find(x -> x>kk, zz[tt_])
                        zz[tt_][idx] -= 1

                    end

                    nn = del_column(nn, kk)
                    zz2table = del_column(zz2table, kk)
                    rcrp.KK -= 1
                end
            end



            ##########################
            ##    resampling kk    ##
            ##########################

            kk_idx_precur = find(x -> x>0, zz2table[tt, :])

            ## compute p(z_{t+1} | z_t)
            kk_idx_curnex = find(x -> x>0, sum(zz2table[tt:tt+1, :], 1))
            KK_curnex = length(kk_idx_curnex) + 1

            L1 = zeros(Float64, rcrp.KK)
            for kk in kk_idx_curnex
                L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + rcrp.aa[tt]/KK_curnex))
            end


            pp = fill(-Inf, rcrp.KK+1)
            for kk in kk_idx_precur
                L1_kk_old = L1[kk]
                L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + 1 + rcrp.aa[tt]/KK_curnex))

                pp[kk] = log(nn[tt, kk]) + logpredictive(components[tt][zz2table[tt, kk]], xx[tt][ii]) + sum(L1) - L2

                L1[kk] = L1_kk_old
            end
            pp[rcrp.KK+1] = log(rcrp.aa[tt]) + logpredictive(rcrp.component, xx[tt][ii]) + sum(L1) - L2


            # for kk in kk_idx_curnex
            #     if !(kk in kk_idx_precur)
            #         L1_kk_old = L1[kk]
            #         L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + 1 + rcrp.aa[tt]/KK_curnex))
            #         pp[kk] = sum(L1) - L2
            #         L1[kk] = L1_kk_old
            #     end
            # end

            lognormalize!(pp)
            kk = sample(pp)


            if kk == rcrp.KK+1
                push!(components[tt], deepcopy(rcrp.component))
                nn = add_column(nn)
                zz2table = add_column(zz2table)
                rcrp.KK += 1
                
                K_t = maximum(zz2table[tt, :])
                zz2table[tt, kk] = K_t + 1
            end

            zz[tt][ii] = kk
            nn[tt, kk] += 1
            additem!(components[tt][zz2table[tt, kk]], xx[tt][ii])
            log_likelihood += loglikelihood(components[tt][zz2table[tt, kk]], xx[tt][ii])
        
        end # iterating over observations


        #############################
        ## cleaning the restaurant ##
        #############################
        # remove empty tables and their dishes so they don't get copied
        # to the next day's restaurant

        kk = 1
        while kk <= rcrp.KK
            if nn[tt, kk] == 0
                if zz2table[tt, kk] != 0
                    splice!(components[tt], zz2table[tt, kk])
                    idx = find(x -> x>zz2table[tt, kk], zz2table[tt, :])
                    zz2table[tt, idx] -= 1
                    zz2table[tt, kk] = 0
                end

                if sum(nn[:, kk]) == 0

                    @inbounds for tt_ = 1:TT
                        idx = find(x -> x>kk, zz[tt_])
                        zz[tt_][idx] -= 1

                        k_t = zz2table[tt_, kk]
                        if k_t != 0
                            splice!(components[tt_], k_t)
                            idx = find(x -> x>k_t, zz2table[tt_, :])
                            zz2table[tt_, idx] -= 1
                            zz2table[tt_, kk] = 0
                        end
                    end

                    nn = del_column(nn, kk)
                    zz2table = del_column(zz2table, kk)
                    rcrp.KK -= 1
                end
            end
            kk += 1
        end


        ## resampling hyper-parameter
        if sample_hyperparam
            K_t = length(findnz(zz2table[tt, :])[1])
            sample_hyperparam!(rcrp, tt, N_t[tt], K_t, n_internals)
        end



        ########################################
        ####### epochs between 2 and T-1 #######
        ########################################

        L2 = sum(log(collect(0 : N_t[tt+1]-1) + N_t[tt] + rcrp.aa[tt]))

        @inbounds for tt = 2:TT-1

            # evolving the dishes at tables that are inherited from previous epoch
            @inbounds for kk = 1:rcrp.KK
                if zz2table[tt-1, kk] != 0
                    if zz2table[tt, kk] != 0
                        components[tt][zz2table[tt, kk]] = evolve(components[tt-1][zz2table[tt-1, kk]])
                        components[tt][zz2table[tt, kk]].nn = nn[tt, kk]
                    else
                        K_t = maximum(zz2table[tt, :])
                        push!(components[tt], evolve(components[tt-1][zz2table[tt-1, kk]]))
                        zz2table[tt, kk] = K_t + 1
                        components[tt][zz2table[tt, kk]].nn = 0
                    end
                end
            end
            


            # iterating over observations
            @inbounds for ii = 1:N_t[tt]

                kk = zz[tt][ii]                     # cluster id
                k_t = zz2table[tt, kk]              # table id


                # remove the observation
                delitem!(components[tt][k_t], xx[tt][ii])
                nn[tt, kk] -= 1

                if nn[tt, kk] == 0 && nn[tt-1, kk] == 0
                    zz2table[tt, kk] = 0
                    splice!(components[tt], k_t)

                    idx = find(x -> x>k_t, zz2table[tt, :])
                    zz2table[tt, idx] -= 1

                    if sum(nn[:, kk]) == 0
                        @inbounds for tt_ = 1:TT

                            k_t = zz2table[tt_, kk]
                            if k_t != 0
                                splice!(components[tt_], k_t)
                                idx = find(x -> x>k_t, zz2table[tt, :])
                                zz2table[tt_, idx] -= 1
                                zz2table[tt_, kk] = 0
                            end

                            idx = find(x -> x>kk, zz[tt_])
                            zz[tt_][idx] -= 1

                        end

                        nn = del_column(nn, kk)
                        zz2table = del_column(zz2table, kk)
                        rcrp.KK -= 1
                    end
                end


                ##########################
                ##    resampling kk    ##
                ##########################


                kk_idx_precur = find(x -> x>0, sum(zz2table[tt-1:tt, :], 1))


                # compute p(z_{t+1} | z_t)
                kk_idx_curnex = find(x -> x>0, sum(zz2table[tt-1:tt+1, :], 1))
                KK_curnex = length(kk_idx_curnex) + 1

                L1 = zeros(Float64, rcrp.KK)
                for kk in kk_idx_curnex
                    L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + rcrp.aa[tt]/KK_curnex))
                end


                pp = fill(-Inf, rcrp.KK+1)
                for kk in kk_idx_precur
                    L1_kk_old = L1[kk]
                    L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + 1 + rcrp.aa[tt]/KK_curnex))

                    pp[kk] = log(nn[tt-1, kk] + nn[tt, kk]) + logpredictive(components[tt][zz2table[tt, kk]], xx[tt][ii]) + sum(L1) - L2

                    L1[kk] = L1_kk_old
                end
                pp[rcrp.KK+1] = log(rcrp.aa[tt]) + logpredictive(rcrp.component, xx[tt][ii]) + sum(L1) - L2

                # for kk in kk_idx_curnex
                #     if !(kk in kk_idx_precur)
                #         L1_kk_old = L1[kk]
                #         L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + 1 + rcrp.aa[tt]/KK_curnex))
                #         pp[kk] = sum(L1) - L2
                #         L1[kk] = L1_kk_old
                #     end
                # end

                lognormalize!(pp)
                kk = sample(pp)

            
                if kk == rcrp.KK+1
                    push!(components[tt], deepcopy(rcrp.component))
                    nn = add_column(nn)
                    zz2table = add_column(zz2table)
                    rcrp.KK += 1

                    K_t = maximum(zz2table[tt, :])
                    zz2table[tt, kk] = K_t + 1
                end

                zz[tt][ii] = kk
                nn[tt, kk] += 1
                additem!(components[tt][zz2table[tt, kk]], xx[tt][ii])
                log_likelihood += loglikelihood(components[tt][zz2table[tt, kk]], xx[tt][ii])
            end # iterating over observations

            # removing empty tables and their dishes
            kk = 1
            while kk <= rcrp.KK

                if nn[tt, kk] == 0

                    # remove any table which has no customer so it
                    # doesn't get copied to the next restaurant.
                    # Remove its dish too.
                    if zz2table[tt, kk] != 0
                        splice!(components[tt], zz2table[tt, kk])
                        idx = find(x -> x>zz2table[tt, kk], zz2table[tt, :])
                        zz2table[tt, idx] -= 1
                        zz2table[tt, kk] = 0
                    end

                    if sum(nn[:, kk]) == 0

                        @inbounds for tt_ = 1:TT
                            idx = find(x -> x>kk, zz[tt_])
                            zz[tt_][idx] -= 1

                            k_t = zz2table[tt_, kk]
                            if k_t != 0
                                splice!(components[tt_], k_t)
                                idx = find(x -> x>k_t, zz2table[tt_, :])
                                zz2table[tt_, idx] -= 1
                                zz2table[tt_, kk] = 0
                            end
                        end

                        nn = del_column(nn, kk)
                        zz2table = del_column(zz2table, kk)
                        rcrp.KK -= 1
                    end
                end
                kk += 1
            end

            # if table kk is empty at time t, treat table kk at time t+1 as a new cluster
            kk = 1
            while kk <= rcrp.KK
                if zz2table[tt, kk] == 0 && zz2table[tt+1, kk] != 0 && sum(zz2table[1:tt, kk]) != 0
                    nn = add_column(nn)
                    zz2table = add_column(zz2table)

                    @inbounds for tt_ = tt+1:TT
                        nn[tt_, rcrp.KK+1] = nn[tt_, kk]
                        zz2table[tt_, rcrp.KK+1] = zz2table[tt_, kk]

                        nn[tt_, kk] = 0
                        zz2table[tt_, kk] = 0

                        idx = find(x -> x==kk, zz[tt_])
                        zz[tt_][idx] = rcrp.KK+1
                    end

                    rcrp.KK += 1
                end
                kk += 1
            end


            ## resampling hyper-parameter

            if sample_hyperparam
                K_t = length(findnz(zz2table[tt, :])[1])
                sample_hyperparam!(rcrp, tt, N_t[tt], K_t, n_internals)
            end
        end


        ####################################
        ############ last epoch ############
        ####################################

        tt = TT

        # evolving the dishes at tables that are inherited from previous epoch
        @inbounds for kk = 1:rcrp.KK
            if zz2table[tt-1, kk] != 0
                if zz2table[tt, kk] != 0
                    components[tt][zz2table[tt, kk]] = evolve(components[tt-1][zz2table[tt-1, kk]])
                    components[tt][zz2table[tt, kk]].nn = nn[tt, kk]
                else
                    KK_curr = maximum(zz2table[tt, :])
                    push!(components[tt], evolve(components[tt-1][zz2table[tt-1, kk]]))
                    zz2table[tt, kk] = KK_curr + 1
                    components[tt][zz2table[tt, kk]].nn = 0
                end
            end
        end

        # iterating over observations
        @inbounds for ii = 1:N_t[TT]

            kk = zz[tt][ii]
            k_t = zz2table[tt, kk]

            delitem!(components[tt][k_t], xx[tt][ii])
            nn[tt, kk] -= 1

            if nn[tt, kk] == 0 && nn[tt-1, kk] == 0
                zz2table[tt, kk] = 0
                splice!(components[tt], k_t)

                idx = find(x -> x>k_t, zz2table[tt, :])
                zz2table[tt, idx] -= 1

                if sum(nn[:, kk]) == 0
                    nn = del_column(nn, kk)
                    zz2table = del_column(zz2table, kk)
                    rcrp.KK -= 1

                    @inbounds for tt_ = 1:TT
                        idx = find(x -> x>kk, zz[tt_])
                        zz[tt_][idx] -= 1
                    end
                end
            end



            ##########################
            ##    resampling kk    ##
            ##########################

            kk_idx_precur = find(x -> x>0, sum(zz2table[tt-1:tt, :], 1))

            pp = fill(-Inf, rcrp.KK+1)

            for kk in kk_idx_precur
                pp[kk] = log(nn[tt-1, kk] + nn[tt, kk]) + logpredictive(components[tt][zz2table[tt, kk]], xx[tt][ii])
            end
            pp[rcrp.KK+1] = log(rcrp.aa[tt]) + logpredictive(rcrp.component, xx[tt][ii])

            lognormalize!(pp)
            kk = sample(pp)


            if kk == rcrp.KK+1
                push!(components[tt], deepcopy(rcrp.component))
                nn = add_column(nn)
                zz2table = add_column(zz2table)
                rcrp.KK += 1
                
                K_t = maximum(zz2table[tt, :])
                zz2table[tt, kk] = K_t + 1
            end            

            zz[tt][ii] = kk
            nn[tt, kk] += 1
            additem!(components[tt][zz2table[tt, kk]], xx[tt][ii])
            log_likelihood += loglikelihood(components[tt][zz2table[tt, kk]], xx[tt][ii])

        end # end of iterating over observations

        # removing empty tables and their dishes
        kk = 1
        while kk <= rcrp.KK

            if nn[tt, kk] == 0

                # remove any table which has no customer so it
                # doesn't get copied to the next restaurant.
                # Remove its dish too.
                if zz2table[tt, kk] != 0
                    splice!(components[tt], zz2table[tt, kk])
                    idx = find(x -> x>zz2table[tt, kk], zz2table[tt, :])
                    zz2table[tt, idx] -= 1
                    zz2table[tt, kk] = 0
                end

                if sum(nn[:, kk]) == 0

                    for tt_ = 1:TT
                        idx = find(x -> x>kk, zz[tt_])
                        zz[tt_][idx] -= 1

                        k_t = zz2table[tt_, kk]
                        if k_t != 0
                            splice!(components[tt_], k_t)
                            idx = find(x -> x>k_t, zz2table[tt_, :])
                            zz2table[tt_, idx] -= 1
                            zz2table[tt_, kk] = 0
                        end
                    end

                    nn = del_column(nn, kk)
                    zz2table = del_column(zz2table, kk)
                    rcrp.KK -= 1
                end
            end
            kk += 1
        end

        ### resampling hyper-parameter ###
        # K_t = length(findnz(zz2table[tt, :])[1])
        # sample_hyperparam!(rcrp, tt, N_t[tt], K_t, n_internals)

        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 && iteration > n_burnins
            sample_n = convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[sample_n] = rcrp.KK
            KK_zz_dict[rcrp.KK] = deepcopy(zz)
            if (sample_n % store_every) == 0
                storesample(rcrp, KK_list, KK_zz_dict, n_burnins, n_lags, sample_n, filename)
            end
        end

    end # iteration
    KK_list, KK_zz_dict
end # function




function posterior{T1, T2}(
    rcrp::RCRP{T1},
    xx::Vector{Vector{T2}},
    KK_zz_dict::Dict{Int, Vector{Vector{Int}}},
    KK::Int)

    TT = length(xx)
    N_t = Array(Int, TT)
    for tt = 1:TT
        N_t[tt] = length(xx[tt])
    end

    nn = zeros(Int, TT, KK)
    zz = KK_zz_dict[KK]

    zz2table = zeros(Int, TT, KK)
    k_t = 1
    for kk = 1:KK
        for tt = 1:TT
            if length(find(x -> x==kk, zz[tt])) != 0
                zz2table[tt, kk] = maximum(zz2table[tt, :]) + 1
            end
        end
    end


    components = Array(Vector{typeof(rcrp.component)}, TT)
    for tt = 1:TT
        K_t = maximum(zz2table[tt, :])
        components[tt] = Array(typeof(rcrp.component), K_t)
        for k_t = 1:K_t
            components[tt][k_t] = deepcopy(rcrp.component)
        end
    end

    for tt = 1:TT
        for ii = 1:N_t[tt]
            nn[tt, zz[tt][ii]] += 1
            k_t = zz2table[tt, zz[tt][ii]]
            additem!(components[tt][k_t], xx[tt][ii])
        end
    end

    pos_components = Array(Vector{typeof(posterior(rcrp.component))}, TT)
    for tt = 1:TT
        K_t = maximum(zz2table[tt, :])        
        pos_components[tt] = Array(typeof(posterior(rcrp.component)), K_t)
        for k_t = 1:K_t
            pos_components[tt][k_t] = posterior(components[tt][k_t])
        end
    end

    return(pos_components, nn, zz2table)
end

