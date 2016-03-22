
#=
RCRP.jl

115/03/2016
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
    KK_dict::Dict{Int, Vector{Vector{Int}}},
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
        "KK_dict", KK_dict,
        "n_burnins", n_burnins, "n_lags", n_lags, "sample_n", sample_n)
end



function sample_hyperparam!(rcrp::RCRP, tt::Int, n::Int, KK::Int, iters::Int)

    for n = 1:iters
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
    KK_dict::Dict{Int, Vector{Vector{Int}}}=Dict{Int, Vector{Vector{Int}}}())




    TT = length(xx)
    N_t = Array(Int, TT)
    for tt = 1:TT
        N_t[tt] = length(xx[tt])
    end

    nn = zeros(Int, TT, rcrp.KK)
    for tt = 1:TT
        for ii = 1:N_t[tt]
            nn[tt, zz[tt][ii]] += 1
        end
    end


    if length(KK_list) == 0
        n_samples_old = 0
        KK_list = zeros(Int, n_samples)
        KK_dict = Dict{Int, Vector{Vector{Int}}}()
    else
        n_samples_old = length(KK_list)        
        KK_list = vcat(KK_list, zeros(Int, n_samples))
    end


    # make sure there is no inactive chain throughout all epochs
    kk = 1
    while kk <= rcrp.KK
        if sum(nn[:, kk]) == 0
            println("\tcomponent $kk is inactive")
            nn = del_column(nn, kk)

            for tt = 1:TT
                idx = find(x -> x>kk, zz[tt])
                zz[tt][idx] -= 1
            end

            rcrp.KK -= 1
        end
        kk += 1
    end



    ## initializing the model ##
    components = Array(typeof(rcrp.component), rcrp.KK)
    for kk = 1:rcrp.KK
        components[kk] = deepcopy(rcrp.component)
    end

    log_likelihood = 0.0
    tic()
    for tt = 1:TT
        for ii = 1:N_t[tt]
            additem!(components[zz[tt][ii]], xx[tt][ii])
            log_likelihood += loglikelihood(components[zz[tt][ii]], xx[tt][ii])
        end
    end
    KK_list[1] = rcrp.KK
    elapsed_time = toq()




    ############################
    ##          MCMC          ##
    ############################
    n_iterations = n_burnins + (n_samples)*(n_lags+1)

    for iteration = 1:n_iterations

        # verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, aa=%.2f, time=%.2f, likelihood=%.2f", iteration, rcrp.KK,
            indmax(hist(KK_list, 0.:maximum(KK_list)+0.5)[2]), rcrp.aa[1], elapsed_time, log_likelihood))



        ##########################
        ##     first epoch      ##
        ##########################
        tt = 1
        log_likelihood = 0.0
        tic()

        L2 = sum(log(collect(0 : N_t[tt+1]-1) + N_t[tt] + rcrp.aa[tt]))
        for ii = randperm(N_t[tt])

            kk  = zz[tt][ii]
            delitem!(components[kk], xx[tt][ii])
            nn[tt, kk] -= 1

            if sum(nn[:, kk]) == 0
                for tt_ = 1:TT
                    idx = find(x -> x>kk, zz[tt_])
                    zz[tt_][idx] -= 1
                end

                splice!(components, kk)
                nn = del_column(nn, kk)
                rcrp.KK -= 1
            end


            kk_idx_precur = find(x -> x>0, nn[tt, :])
            kk_idx_curnex = find(x -> x>0, sum(nn[tt:tt+1, :], 1))
            KK_curnex = length(kk_idx_curnex) + 1

            L1 = zeros(Float64, rcrp.KK)
            for kk in kk_idx_curnex
                L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + rcrp.aa[tt]/KK_curnex))
            end


            # resample kk
            pp = fill(-Inf, rcrp.KK+1)
            for kk in kk_idx_precur
                L1_kk_old = L1[kk]
                L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + 1 + rcrp.aa[tt]/KK_curnex))

                pp[kk] = log(nn[tt, kk]) + logpredictive(components[kk], xx[tt][ii]) + sum(L1) - L2

                L1[kk] = L1_kk_old
            end
            pp[rcrp.KK+1] = log(rcrp.aa[tt]) + logpredictive(rcrp.component, xx[tt][ii]) + sum(L1) - L2

            lognormalize!(pp)
            kk = sample(pp)

            if kk == rcrp.KK+1
                push!(components, deepcopy(rcrp.component))
                nn = add_column(nn)
                rcrp.KK += 1
            end

            zz[tt][ii] = kk
            nn[tt, kk] += 1
            additem!(components[kk], xx[tt][ii])
            log_likelihood += loglikelihood(components[kk], xx[tt][ii])

        end # ii

        if sample_hyperparam
            KK_tt = length(findnz(nn[tt, :])[1])
            sample_hyperparam!(rcrp, tt, N_t[tt], KK_tt, n_internals)
        end




        #########################
        ## epochs 2 < tt < T-1 ##
        #########################
        for tt = 2:TT-1


            L2 = sum(log(collect(0 : N_t[tt+1]-1) + N_t[tt] + rcrp.aa[tt]))

            for ii = randperm(N_t[tt])

                kk = zz[tt][ii]
                delitem!(components[kk], xx[tt][ii])
                nn[tt, kk] -= 1

                if sum(nn[:, kk]) == 0
                    for tt_ = 1:TT
                        idx = find(x -> x>kk, zz[tt_])
                        zz[tt_][idx] -= 1
                    end

                    splice!(components, kk)
                    nn = del_column(nn, kk)
                    rcrp.KK -= 1
                end



                kk_idx_precur = find(x -> x>0, sum(nn[tt-1:tt, :], 1))
                kk_idx_curnex = find(x -> x>0, sum(nn[tt-1:tt+1, :], 1))
                KK_curnex = length(kk_idx_curnex) + 1

                L1 = zeros(Float64, rcrp.KK)
                for kk in kk_idx_curnex
                    L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + rcrp.aa[tt]/KK_curnex))
                end

                # resample kk
                pp = fill(-Inf, rcrp.KK+1)
                for kk in kk_idx_precur
                    L1_kk_old = L1[kk]
                    L1[kk] = sum(log(collect(0:nn[tt+1, kk]-1) + nn[tt, kk] + 1 + rcrp.aa[tt]/KK_curnex))

                    pp[kk] = log(nn[tt-1, kk] + nn[tt, kk]) + logpredictive(components[kk], xx[tt][ii]) + sum(L1) - L2

                    L1[kk] = L1_kk_old
                end
                pp[rcrp.KK+1] = log(rcrp.aa[tt]) + logpredictive(rcrp.component, xx[tt][ii]) + sum(L1) - L2

                lognormalize!(pp)
                kk = sample(pp)


                if kk == rcrp.KK+1
                    push!(components, deepcopy(rcrp.component))
                    nn = add_column(nn)
                    rcrp.KK += 1
                end

                zz[tt][ii] = kk
                nn[tt, kk] += 1
                additem!(components[kk], xx[tt][ii])
                log_likelihood += loglikelihood(components[kk], xx[tt][ii])
            end # ii


            # if table kk is empty at time t, treat table kk at time t+1 as a new cluster
            kk = 1
            while kk <= rcrp.KK
                if nn[tt, kk] == 0 && nn[tt+1, kk] != 0 && sum(nn[1:tt, kk]) != 0

                    push!(components, deepcopy(rcrp.component))
                    nn = add_column(nn)

                    for tt_ = tt+1:TT
                        idx = find(x -> x==kk, zz[tt_])
                        zz[tt_][idx] = rcrp.KK+1
                        
                        for idid in idx
                            delitem!(components[kk], xx[tt_][idid])
                            additem!(components[rcrp.KK+1], xx[tt_][idid])                            
                        end

                        nn[tt_, rcrp.KK+1] = nn[tt_, kk]
                        nn[tt_, kk] = 0
                    end
                    rcrp.KK += 1
                end
                kk += 1
            end


            if sample_hyperparam
                KK_tt = length(findnz(nn[tt, :])[1])
                sample_hyperparam!(rcrp, tt, N_t[tt], KK_tt, n_internals)
            end

        end # tt



        ########################
        ##     last epoch     ##
        ########################
        tt = TT

        for ii = randperm(N_t[tt])

            kk = zz[tt][ii]

            delitem!(components[kk], xx[tt][ii])
            nn[tt, kk] -= 1

            if sum(nn[:, kk]) == 0
                for tt_ = 1:TT
                    idx = find(x -> x>kk, zz[tt_])
                    zz[tt_][idx] -= 1
                end

                splice!(components, kk)
                nn = del_column(nn, kk)
                rcrp.KK -= 1
            end



            kk_idx_precur = find(x -> x>0, sum(nn[tt-1:tt, :], 1))

            # resdample kk
            pp = fill(-Inf, rcrp.KK+1)
            for kk in kk_idx_precur
                pp[kk] = log(nn[tt-1, kk] + nn[tt, kk]) + logpredictive(components[kk], xx[tt][ii])
            end
            pp[rcrp.KK+1] = log(rcrp.aa[tt]) + logpredictive(rcrp.component, xx[tt][ii])

            lognormalize!(pp)
            kk = sample(pp)



            if kk == rcrp.KK+1
                push!(components, deepcopy(rcrp.component))
                nn = add_column(nn)
                rcrp.KK += 1
            end

            zz[tt][ii] = kk
            nn[tt, kk] += 1
            additem!(components[kk], xx[tt][ii])
            log_likelihood += loglikelihood(components[kk], xx[tt][ii])

        end # ii

        if sample_hyperparam
            KK_tt = length(findnz(nn[tt, :])[1])
            sample_hyperparam!(rcrp, tt, N_t[tt], KK_tt, n_internals)
        end



        elapsed_time = toq()
        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 && iteration > n_burnins
            sample_n = n_samples_old + convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[sample_n] = rcrp.KK
            KK_dict[rcrp.KK] = deepcopy(zz)
            if (sample_n % store_every) == 0
                storesample(rcrp, KK_list, KK_dict, n_burnins, n_lags, sample_n, filename)
            end
        end


    end # iteration
    KK_list, KK_dict
end



function posterior{T1, T2}(
    rcrp::RCRP{T1},
    xx::Vector{Vector{T2}},
    KK_dict::Dict{Int, Vector{Vector{Int}}},
    KK::Int)

    TT = length(xx)
    N_t = Array(Int, TT)
    for tt = 1:TT
        N_t[tt] = length(xx[tt])
    end

    nn = zeros(Int, TT, KK)
    zz = KK_dict[KK]

    components = Array(typeof(rcrp.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(rcrp.component)
    end


    for tt = 1:TT
        for ii = 1:N_t[tt]
            nn[tt, zz[tt][ii]] += 1
            additem!(components[zz[tt][ii]], xx[tt][ii])
        end
    end

    pos_components = Array(typeof(posterior(rcrp.component)), KK)
    for kk = 1:KK
        pos_components[kk] = posterior(components[kk])
    end

    return(pos_components, nn)
end