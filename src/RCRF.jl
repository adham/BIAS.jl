#=
RCRF.jl

Adham Beyki, odinay@gmail.com
22/03/2016

NOTE
    The hyperparam resampling is just an approximate as the conditional distirbution of
    Z isn't a DP. We just use it to approximate the hyperparams and then set it to false

=#

####################################################
##     Recurrent Chinese Restaurant Franchise     ##
####################################################
type RCRF{T} <: hMixtureModel
    component::T
    KK::Int
    gg::Vector{Float64}
    g1::Float64
    g2::Float64
    aa::Vector{Float64}
    a1::Float64
    a2::Float64

    RCRF{T}(c::T,
            KK::Int,
            gg::Vector{Float64}, g1::Float64, g2::Float64,
            aa::Vector{Float64}, a1::Float64, a2::Float64) = new(c, KK, gg, g1, g2, aa, a1, a2)
end

RCRF{T}(c::T,
        KK::Int,
        gg::Real, g1::Real, g2::Real,
        aa::Real, a1::Real, a2::Real,
        TT::Int) = RCRF{typeof(c)}(c, KK,
        fill(convert(Float64, gg), TT), convert(Float64, g1), convert(Float64, g2),
        fill(convert(Float64, aa), TT), convert(Float64, a1), convert(Float64, a2))


function Base.show(io::IO, rcrf::RCRF)
    println(io, "Recurrent Chinese Restaurant Franchise Mixture Model with $(rcrf.KK) $(typeof(rcrf.component)) components")
end



type RCRFSample
    njt::Vector{Vector{Vector{Int}}}
    kjt::Vector{Vector{Vector{Int}}}
    zz::Vector{Vector{Vector{Int}}}

    RCRFSample(njt::Vector{Vector{Vector{Int}}}, kjt::Vector{Vector{Vector{Int}}}, zz::Vector{Vector{Vector{Int}}}) = new(njt, kjt, zz)
end


function storesample{T}(
    rcrf::RCRF{T},
    KK_list::Vector{Int},
    KK_dict::Dict{Int, Vector{Vector{Vector{Int}}}},
    n_burnins::Int, n_lags::Int, sample_n::Int,
    filename::ASCIIString)

    println("\nstoring on disk...\n")
    if endswith(filename, "_")
        dummy_filename = string(filename, sample_n, ".jld")
    else
        dummy_filename = string(filename, "_", sample_n, ".jld")
    end

    JLD.save(dummy_filename,
        "rcrf", rcrf,
        "KK_list", KK_list,
        "KK_dict", KK_dict,
        "n_burnins", n_burnins, "n_lags", n_lags, "sample_n", sample_n)
end




function sample_hyperparam!(rcrf::RCRF, tt::Int, n_group_j::Vector{Int}, m::Int, KK::Int, n_internals::Int)
    #= NOTE
    resampling the group level concentration parameter α0 using auxiliary variables
    w and s, Eq. 50 Teh etal 04 the Gamma distribution in Eq.50 is expressed using
    shape and rate. We have to rescale them to shape and scale representation to be
    able to use standard random gamma functions in Julia to draw from it. Also:
    Gamma(a, 1/b) = Gamma(a) / b
    =#

    for iteration = 1:n_internals
        n_groups = length(n_group_j)
        w = zeros(Float64, n_groups)
        for jj = 1:n_groups
            w[jj] = rand(Distributions.Beta(rcrf.aa[tt]+1, n_group_j[jj]))
        end
        p = n_group_j / rcrf.aa[tt]
        p ./= (p+1.0)

        s = zeros(Int, n_groups)
        for jj = 1:n_groups
            s[jj] = rand(Distributions.Binomial(1, p[jj]))
        end

        aa_shape = rcrf.a1 + m - sum(s)
        aa_rate  = rcrf.a2 - sum(log(w))
        rcrf.aa[tt] = rand(Distributions.Gamma(aa_shape)) / aa_rate


        # resampling the top level concentration parameter γ, Escobar and West 95
        eta = rand(Distributions.Beta(rcrf.gg[tt]+1, m))
        pi_eta = 1 / (1 + (m*(rcrf.g2 - log(eta))) / (rcrf.g1 + KK - 1))

        if rand() < pi_eta
            rcrf.gg[tt] = rand(Distributions.Gamma(rcrf.g1+KK)) / (rcrf.g2-log(eta))
        else
            rcrf.gg[tt] = rand(Distributions.Gamma(rcrf.g1+KK-1)) / (rcrf.g2-log(eta))
        end
    end
end # sample_hyperparam




function RCRF_gibbs_sampler!{T1, T2}(
    rcrf::RCRF{T1},
    xx::Vector{Vector{Vector{T2}}},
    zz::Vector{Vector{Vector{Int}}},
    n_burnins::Int, n_lags::Int, n_samples::Int,
    sample_hyperparam::Bool=true, n_internals::Int=10,
    store_every::Int=100, filename::ASCIIString="RCRF_results_",
    join_tables::Bool=true,
    KK_list::Vector{Int}=Int[],
    KK_dict::Dict{Int, Vector{Vector{Vector{Int}}}}=Dict{Int, Vector{Vector{Vector{Int}}}}())


    n_iterations = n_burnins + (n_samples)*(n_lags+1)
    TT           = length(xx)                           # number of epochs
    n_groups     = [length(xx[tt]) for tt=1:TT]         # number of restaurants per epoch
    tji = Array(Vector{Vector{Int}}, TT)                # tji[tt][jj][ii]  represents the table that customer ii at restaurant jj at epoch tt occupies
    njt = Array(Vector{Vector{Int}}, TT)                # njt[tt][jj][tbl] represents the number of customers sitting at table tbl, in restaurant jj at epoch tt
    kjt = Array(Vector{Vector{Int}}, TT)                # kjt[tt][jj][tbl] represents the dish being served at table tbl, restaurant jj at epoch tt
    mm = zeros(Int, TT, rcrf.KK)                        # mm[tt, kk] how many tables at epoch tt serve dish kk


    # number of customers per restaurant per epoch
    n_group_j = Array(Vector{Int}, TT)
    for tt = 1:TT
        n_group_j[tt] = zeros(Int, n_groups[tt])
        for jj = 1:n_groups[tt]
            n_group_j[tt][jj] = length(xx[tt][jj])
        end
    end


    # instantiating the conjugate components
    components = Array(typeof(rcrf.component), rcrf.KK)
    for kk = 1:rcrf.KK
        components[kk] = deepcopy(rcrf.component)
    end

    if length(KK_list) == 0
        n_samples_old = 0
        KK_list = zeros(Int, n_samples)
        KK_dict = Dict{Int, Vector{Vector{Vector{Int}}}}()
    else
        n_samples_old = length(KK_list)
        KK_list = vcat(KK_list, zeros(Int, n_samples))
    end


    ##############################
    ##  Initializing the model  ##
    ##############################
    log_likelihood = 0.0
    tic()

    print_with_color(:blue, "\nInitializing the model\n\n")

    for tt = 1:TT
        kjt[tt] = Array(Vector{Int}, n_groups[tt])
        tji[tt] = Array(Vector{Int}, n_groups[tt])
        njt[tt] = Array(Vector{Int}, n_groups[tt])

        for jj = 1:n_groups[tt]
            kjt[tt][jj] = unique(zz[tt][jj])
            n_tbls = length(kjt[tt][jj])

            tji[tt][jj] = zeros(Int, n_group_j[tt][jj])
            njt[tt][jj] = zeros(Int, n_tbls)

            for ii = 1:n_group_j[tt][jj]
                kk = zz[tt][jj][ii]
                additem!(components[kk], xx[tt][jj][ii])
                log_likelihood += loglikelihood(components[kk], xx[tt][jj][ii])

                tbl = findfirst(kjt[tt][jj], kk)
                tji[tt][jj][ii] = tbl
                njt[tt][jj][tbl] += 1
            end # for ii
        end # for jj
    end # for tt


    for tt = 1:TT
        for jj = 1:n_groups[tt]
            for kk in kjt[tt][jj]
                mm[tt, kk] += 1
            end
        end
    end
    KK_list[1] = rcrf.KK
    elapsed_time = toq()


    #################################
    ##   starting the MCCM chain   ##
    #################################
    for iteration = 1:n_iterations

        # verbose
        if iteration < n_burnins
            print_with_color(:blue, "Burning... ")
        end
        println(@sprintf("iteration: %d, KK=%d, KK mode=%d, aa=%.2f, gg=%.2f, time=%.2f, likelihood=%.2f", iteration,
            rcrf.KK, indmax(hist(KK_list, 0.:maximum(KK_list)+0.5)[2]), rcrf.aa[1], rcrf.gg[1], elapsed_time, log_likelihood))

        if iteration > n_burnins
            sample_hyperparam = false
        end

        #############################
        ##         epoch 1         ##
        #############################
        tic()
        tt = 1
        log_likelihood = 0.0

        ## sample the tables for customers ##
        for jj = 1:n_groups[tt]
            n_tbls = length(kjt[tt][jj])

            for ii = 1:n_group_j[tt][jj]

                kk = zz[tt][jj][ii]
                tbl = tji[tt][jj][ii]

                # remove the effect of point xx[tt][jj][ii]
                delitem!(components[kk], xx[tt][jj][ii])
                njt[tt][jj][tbl] -= 1

                # if its table becomes empty, remove the table
                if njt[tt][jj][tbl] == 0

                    # the dish which is served at tbl
                    kk = kjt[tt][jj][tbl]
                    n_tbls -= 1

                    # remove the table
                    splice!(kjt[tt][jj], tbl)
                    splice!(njt[tt][jj], tbl)

                    # shift the table ids
                    idx = find(x -> x>tbl, tji[tt][jj])
                    tji[tt][jj][idx] -= 1

                    # remove the table dish
                    mm[tt, kk] -= 1

                    if sum(mm[:, kk]) == 0
                        for tt_ = 1:TT
                            for jj_ = 1:n_groups[tt_]
                                idx = find(x -> x>kk, zz[tt_][jj_])
                                zz[tt_][jj_][idx] -= 1

                                tbl_idx = unique(tji[tt_][jj_][idx])
                                kjt[tt_][jj_][tbl_idx] -= 1
                            end
                        end

                        splice!(components, kk)
                        mm = del_column(mm, kk)
                        rcrf.KK -= 1
                    end
                end # if njt == 0

                # resample the table, follows CRF
                pp = zeros(Float64, n_tbls+1)
                for tbl = 1:n_tbls
                    kk = kjt[tt][jj][tbl]
                    pp[tbl] = log(njt[tt][jj][tbl]) + logpredictive(components[kk], xx[tt][jj][ii])
                end
                pp[n_tbls+1] = log(rcrf.aa[tt]) + logpredictive(rcrf.component, xx[tt][jj][ii])


                lognormalize!(pp)
                tbl = sample(pp)


                # if an empty table is selected
                if tbl == n_tbls+1

                    n_tbls += 1

                    push!(kjt[tt][jj], 0)
                    push!(njt[tt][jj], 0)

                    # sample the dish for the newly instantiated table
                    kk_idx_precur = find(x -> x>0, mm[tt, :])
                    kk_idx_curnex = find(x -> x>0, sum(mm[tt:tt+1, :], 1))
                    KK_curnex = length(kk_idx_curnex) + 1

                    L1 = zeros(Float64, rcrf.KK)
                    for kk in kk_idx_curnex
                        L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + rcrf.gg[tt]/KK_curnex))
                    end
                    L2 = sum(log(collect(0:(sum(mm[tt+1, :])-1)) + sum(mm[tt, :]) + rcrf.gg[tt]))


                    pp = fill(-Inf, rcrf.KK+1)
                    for kk in kk_idx_precur
                        L1_kk_old = L1[kk]
                        L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + 1 + rcrf.aa[tt]/KK_curnex))

                        pp[kk] = log(mm[tt, kk]) + logpredictive(components[kk], xx[tt][jj][ii]) + sum(L1) - L2
                        L1[kk] = L1_kk_old
                    end
                    pp[rcrf.KK+1] = log(rcrf.gg[tt]) + logpredictive(rcrf.component, xx[tt][jj][ii]) + sum(L1) - L2


                    lognormalize!(pp)
                    kk = sample(pp)

                    # if the selected dish is new
                    if kk == rcrf.KK+1
                        push!(components, deepcopy(rcrf.component))
                        rcrf.KK += 1
                        mm = add_column(mm)
                    end

                    kjt[tt][jj][tbl] = kk
                    mm[tt, kk] += 1
                end # if tbl is new

                kk = kjt[tt][jj][tbl]
                tji[tt][jj][ii] = tbl
                njt[tt][jj][tbl] += 1
                additem!(components[kk], xx[tt][jj][ii])
                zz[tt][jj][ii] = kk
            end # for ii
        end # for jj

        if join_tables
            for jj=1:n_groups[tt]
                kk_list = unique(kjt[tt][jj])
                if length(kk_list) != length(kjt[tt][jj])
                    new_kjt = zeros(Int, length(kk_list))
                    new_njt = zeros(Int, length(kk_list))
                    ll = 1

                    for unq_kk in kk_list
                        idx = find(x -> x==unq_kk, kjt[tt][jj])
                        new_kjt[ll] = unq_kk
                        new_njt[ll] = sum(njt[tt][jj][idx])

                        for idid in idx
                            tbl_idx = find(x -> x==idid, tji[tt][jj])
                            tji[tt][jj][tbl_idx] = ll
                        end

                        mm[tt, unq_kk] -= (length(idx) - 1)

                        ll += 1
                    end
                    kjt[tt][jj] = new_kjt
                    njt[tt][jj] = new_njt
                end
            end
        end

        ## sample the dishes for tables
        for jj = 1:n_groups[tt]

            # number of tables in restaurant jj
            n_tbls = length(kjt[tt][jj])

            for tbl = 1:n_tbls

                kk = kjt[tt][jj][tbl]
                mm[tt, kk] -= 1

                tidx = find(x -> x==tbl, tji[tt][jj])


                # if tbl is the only table at restaurant jj at time tt which serves dish kk,
                # we need to remove the dish from the epoch menu
                if mm[tt, kk] == 0
                    if sum(mm[:, kk]) == 0
                        for tt_ = 1:TT
                            for jj_ = 1:n_groups[tt_]
                                idx = find(x -> x>kk, zz[tt_][jj_])
                                zz[tt_][jj_][idx] -= 1

                                tbl_idx = unique(tji[tt_][jj_][idx])
                                kjt[tt_][jj_][tbl_idx] -= 1
                            end
                        end # for tt_

                        splice!(components, kk)
                        mm = del_column(mm, kk)
                        rcrf.KK -= 1
                    end
                else
                    for ll in tidx
                        delitem!(components[kk], xx[tt][jj][ll])
                    end
                end

                kk_idx_precur = find(x -> x>0, mm[tt, :])
                kk_idx_curnex = find(x -> x>0, sum(mm[tt:tt+1, :], 1))
                KK_curnex = length(kk_idx_curnex) + 1

                L1 = zeros(Float64, rcrf.KK)
                for kk in kk_idx_curnex
                    L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + rcrf.gg[tt]/KK_curnex))
                end
                L2 = sum(log(collect(0:(sum(mm[tt+1, :])-1)) + sum(mm[tt, :]) + rcrf.gg[tt]))

                # resample the dish
                pp = fill(-Inf, rcrf.KK+1)
                for kk in kk_idx_precur
                    L1_kk_old = L1[kk]
                    L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + 1 + rcrf.aa[tt]/KK_curnex))

                    pp[kk] = log(mm[tt, kk]) + sum(L1) - L2
                    for ll in tidx
                        pp[kk] += logpredictive(components[kk], xx[tt][jj][ll])
                    end

                    L1[kk] = L1_kk_old
                end
                pp[rcrf.KK+1] = log(rcrf.gg[tt]) + sum(L1) - L2
                for ll in tidx
                    pp[rcrf.KK+1] += logpredictive(rcrf.component, xx[tt][jj][ll])
                end

                lognormalize!(pp)
                kk = sample(pp)

                if kk == rcrf.KK+1
                    push!(components, deepcopy(rcrf.component))
                    rcrf.KK += 1
                    mm = add_column(mm)
                end

                kjt[tt][jj][tbl] = kk
                for ll in tidx
                    additem!(components[kk], xx[tt][jj][ll])
                end
                mm[tt, kk] += 1
                zz[tt][jj][tidx] = kk
            end # for tbl
        end # for jj

        # resample hyper-parameters
        if sample_hyperparam
            KK_tt = length(findnz(mm[tt, :])[1])
            m = sum([length(kjt[tt][jj]) for jj=1:n_groups[tt]])
            sample_hyperparam!(rcrf, tt, n_group_j[tt], m, KK_tt, n_internals)
        end


        ############################
        ##      epoch 2:TT-1      ##
        ############################
        for tt = 2:TT-1
            for jj = 1:n_groups[tt]
                n_tbls = length(kjt[tt][jj])

                for ii = 1:n_group_j[tt][jj]

                    kk = zz[tt][jj][ii]
                    tbl = tji[tt][jj][ii]

                    delitem!(components[kk], xx[tt][jj][ii])
                    njt[tt][jj][tbl] -= 1

                    if njt[tt][jj][tbl] == 0

                        kk = kjt[tt][jj][tbl]
                        n_tbls -= 1

                        splice!(kjt[tt][jj], tbl)
                        splice!(njt[tt][jj], tbl)

                        idx = find(x -> x>tbl, tji[tt][jj])
                        tji[tt][jj][idx] -= 1

                        mm[tt, kk] -= 1
                        if sum(mm[:, kk]) == 0
                            for tt_ = 1:TT
                                for jj_ = 1:n_groups[tt_]
                                    idx = find(x -> x>kk, zz[tt_][jj_])
                                    zz[tt_][jj_][idx] -= 1

                                    tbl_idx = unique(tji[tt_][jj_][idx])
                                    kjt[tt_][jj_][tbl_idx] -= 1
                                end
                            end # for tt_

                            splice!(components, kk)
                            mm = del_column(mm, kk)
                            rcrf.KK -= 1
                        end # if sum(mm[:, kk])
                    end # if njt


                    pp = zeros(Float64, n_tbls+1)
                    for tbl = 1:n_tbls
                        kk = kjt[tt][jj][tbl]
                        pp[tbl] = log(njt[tt][jj][tbl]) + logpredictive(components[kk], xx[tt][jj][ii])
                    end
                    pp[n_tbls+1] = log(rcrf.aa[tt]) + logpredictive(rcrf.component, xx[tt][jj][ii])


                    lognormalize!(pp)
                    tbl = sample(pp)


                    if tbl == n_tbls+1

                        n_tbls += 1

                        push!(kjt[tt][jj], 0)
                        push!(njt[tt][jj], 0)

                        kk_idx_precur = find(x -> x>0, sum(mm[tt-1:tt, :], 1))
                        kk_idx_curnex = find(x -> x>0, sum(mm[tt:tt+1, :], 1))
                        KK_curnex = length(kk_idx_curnex) + 1

                        L1 = zeros(Float64, rcrf.KK)
                        for kk in kk_idx_curnex
                            L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + rcrf.gg[tt]/KK_curnex))
                        end
                        L2 = sum(log(collect(0:(sum(mm[tt+1, :])-1)) + sum(mm[tt, :]) + rcrf.gg[tt]))


                        pp = fill(-Inf, rcrf.KK+1)
                        for kk in kk_idx_precur
                            L1_kk_old = L1[kk]
                            L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + 1 + rcrf.aa[tt]/KK_curnex))

                            pp[kk] = log(mm[tt-1, kk] + mm[tt, kk]) + logpredictive(components[kk], xx[tt][jj][ii]) + sum(L1) - L2
                            L1[kk] = L1_kk_old
                        end
                        pp[rcrf.KK+1] = log(rcrf.gg[tt]) + logpredictive(rcrf.component, xx[tt][jj][ii]) + sum(L1) - L2


                        lognormalize!(pp)
                        kk = sample(pp)

                        if kk == rcrf.KK+1
                            push!(components, deepcopy(rcrf.component))
                            rcrf.KK += 1
                            mm = add_column(mm)
                        end

                        kjt[tt][jj][tbl] = kk
                        mm[tt, kk] += 1
                    end # if tbl is new

                    kk = kjt[tt][jj][tbl]
                    tji[tt][jj][ii] = tbl
                    njt[tt][jj][tbl] += 1
                    additem!(components[kk], xx[tt][jj][ii])
                    zz[tt][jj][ii] = kk
                end # for ii
            end # for jj

            if join_tables
                for jj=1:n_groups[tt]
                    kk_list = unique(kjt[tt][jj])
                    if length(kk_list) != length(kjt[tt][jj])
                        new_kjt = zeros(Int, length(kk_list))
                        new_njt = zeros(Int, length(kk_list))
                        ll = 1

                        for unq_kk in kk_list
                            idx = find(x -> x==unq_kk, kjt[tt][jj])
                            new_kjt[ll] = unq_kk
                            new_njt[ll] = sum(njt[tt][jj][idx])

                            for idid in idx
                                tbl_idx = find(x -> x==idid, tji[tt][jj])
                                tji[tt][jj][tbl_idx] = ll
                            end

                            mm[tt, unq_kk] -= (length(idx) - 1)

                            ll += 1
                        end
                        kjt[tt][jj] = new_kjt
                        njt[tt][jj] = new_njt
                    end
                end
            end

            for jj = 1:n_groups[tt]

                # number of tables in restaurant jj
                n_tbls = length(kjt[tt][jj])

                for tbl = 1:n_tbls

                    kk = kjt[tt][jj][tbl]
                    mm[tt, kk] -= 1

                    tidx = find(x -> x==tbl, tji[tt][jj])


                    # if tbl is the only table at restaurant jj at time tt which serves dish kk,
                    # we need to remove the dish from the epoch menu
                    if mm[tt, kk] == 0
                        if sum(mm[:, kk]) == 0
                            for tt_ = 1:TT
                                for jj_ = 1:n_groups[tt_]
                                    idx = find(x -> x>kk, zz[tt_][jj_])
                                    zz[tt_][jj_][idx] -= 1

                                    tbl_idx = unique(tji[tt_][jj_][idx])
                                    kjt[tt_][jj_][tbl_idx] -= 1
                                end
                            end # for tt_

                            splice!(components, kk)
                            mm = del_column(mm, kk)
                            rcrf.KK -= 1
                        end
                    else
                        for ll in tidx
                            delitem!(components[kk], xx[tt][jj][ll])
                        end
                    end


                    kk_idx_precur = find(x -> x>0, sum(mm[tt-1:tt, :], 1))
                    kk_idx_curnex = find(x -> x>0, sum(mm[tt:tt+1, :], 1))
                    KK_curnex = length(kk_idx_curnex) + 1

                    L1 = zeros(Float64, rcrf.KK)
                    for kk in kk_idx_curnex
                        L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + rcrf.gg[tt]/KK_curnex))
                    end
                    L2 = sum(log(collect(0:(sum(mm[tt+1, :])-1)) + sum(mm[tt, :]) + rcrf.gg[tt]))

                    # resample the dish
                    pp = fill(-Inf, rcrf.KK+1)
                    for kk in kk_idx_precur
                        L1_kk_old = L1[kk]
                        L1[kk] = sum(log(collect(0:mm[tt+1, kk]-1) + mm[tt, kk] + 1 + rcrf.aa[tt]/KK_curnex))

                        pp[kk] = log(mm[tt-1, kk] + mm[tt, kk]) + sum(L1) - L2
                        for ll in tidx
                            pp[kk] += logpredictive(components[kk], xx[tt][jj][ll])
                        end

                        L1[kk] = L1_kk_old
                    end
                    pp[rcrf.KK+1] = log(rcrf.gg[tt]) + sum(L1) - L2
                    for ll in tidx
                        pp[rcrf.KK+1] += logpredictive(rcrf.component, xx[tt][jj][ll])
                    end

                    lognormalize!(pp)
                    kk = sample(pp)

                    if kk == rcrf.KK+1
                        push!(components, deepcopy(rcrf.component))
                        rcrf.KK += 1
                        mm = add_column(mm)
                    end

                    kjt[tt][jj][tbl] = kk
                    for ll in tidx
                        additem!(components[kk], xx[tt][jj][ll])
                    end
                    mm[tt, kk] += 1
                    zz[tt][jj][tidx] = kk
                end # for tbl
            end # for jj

            # resample hyper-parameters
            if sample_hyperparam
                KK_tt = length(findnz(mm[tt, :])[1])
                m = sum([length(kjt[tt][jj]) for jj=1:n_groups[tt]])
                sample_hyperparam!(rcrf, tt, n_group_j[tt], m, KK_tt, n_internals)
            end

        end # tt



        ############################
        ##        epoch TT        ##
        ############################
        tt = TT
        for jj = 1:n_groups[tt]
            n_tbls = length(kjt[tt][jj])

            for ii = 1:n_group_j[tt][jj]

                kk = zz[tt][jj][ii]
                tbl = tji[tt][jj][ii]

                delitem!(components[kk], xx[tt][jj][ii])
                njt[tt][jj][tbl] -= 1

                if njt[tt][jj][tbl] == 0

                    kk = kjt[tt][jj][tbl]
                    n_tbls -= 1

                    splice!(kjt[tt][jj], tbl)
                    splice!(njt[tt][jj], tbl)

                    idx = find(x -> x>tbl, tji[tt][jj])
                    tji[tt][jj][idx] -= 1

                    mm[tt, kk] -= 1
                    if sum(mm[:, kk]) == 0
                        for tt_ = 1:TT
                            for jj_ = 1:n_groups[tt_]
                                idx = find(x -> x>kk, zz[tt_][jj_])
                                zz[tt_][jj_][idx] -= 1

                                tbl_idx = unique(tji[tt_][jj_][idx])
                                kjt[tt_][jj_][tbl_idx] -= 1
                            end
                        end
                        splice!(components, kk)
                        mm = del_column(mm, kk)
                        rcrf.KK -= 1
                    end
                end # if njt


                pp = zeros(Float64, n_tbls+1)
                for tbl = 1:n_tbls
                    kk = kjt[tt][jj][tbl]
                    pp[tbl] = log(njt[tt][jj][tbl]) + logpredictive(components[kk], xx[tt][jj][ii])
                end
                pp[n_tbls+1] = log(rcrf.aa[tt]) + logpredictive(rcrf.component, xx[tt][jj][ii])


                lognormalize!(pp)
                tbl = sample(pp)

                if tbl == n_tbls+1

                    n_tbls += 1

                    push!(kjt[tt][jj], 0)
                    push!(njt[tt][jj], 0)

                    kk_idx_precur = find(x -> x>0, sum(mm[tt-1:tt, :], 1))

                    pp = fill(-Inf, rcrf.KK+1)
                    for kk in kk_idx_precur
                        pp[kk] = log(mm[tt-1, kk] + mm[tt, kk]) + logpredictive(components[kk], xx[tt][jj][ii])
                    end
                    pp[rcrf.KK+1] = log(rcrf.gg[tt]) + logpredictive(rcrf.component, xx[tt][jj][ii])


                    lognormalize!(pp)
                    kk = sample(pp)

                    # a new dish is served
                    if kk == rcrf.KK + 1
                        push!(components, deepcopy(rcrf.component))
                        rcrf.KK += 1
                        mm = add_column(mm)
                    end

                    kjt[tt][jj][tbl] = kk
                    mm[tt, kk] += 1
                end # if tbl is new

                kk = kjt[tt][jj][tbl]
                tji[tt][jj][ii] = tbl
                njt[tt][jj][tbl] += 1
                additem!(components[kk], xx[tt][jj][ii])
                zz[tt][jj][ii] = kk
            end # for ii
        end # for jj

        if join_tables
            for jj=1:n_groups[tt]
                kk_list = unique(kjt[tt][jj])
                if length(kk_list) != length(kjt[tt][jj])
                    new_kjt = zeros(Int, length(kk_list))
                    new_njt = zeros(Int, length(kk_list))
                    ll = 1

                    for unq_kk in kk_list
                        idx = find(x -> x==unq_kk, kjt[tt][jj])
                        new_kjt[ll] = unq_kk
                        new_njt[ll] = sum(njt[tt][jj][idx])

                        for idid in idx
                            tbl_idx = find(x -> x==idid, tji[tt][jj])
                            tji[tt][jj][tbl_idx] = ll
                        end

                        mm[tt, unq_kk] -= (length(idx) - 1)

                        ll += 1
                    end
                    kjt[tt][jj] = new_kjt
                    njt[tt][jj] = new_njt
                end
            end
        end

        for jj = 1:n_groups[tt]

            # number of tables in restaurant jj
            n_tbls = length(kjt[tt][jj])

            for tbl = 1:n_tbls

                kk = kjt[tt][jj][tbl]
                mm[tt, kk] -= 1

                tidx = find(x -> x==tbl, tji[tt][jj])


                # if tbl is the only table at restaurant jj at time tt which serves dish kk,
                # we need to remove the dish from the epoch menu
                if mm[tt, kk] == 0
                    if sum(mm[:, kk]) == 0
                        for tt_ = 1:TT
                            for jj_ = 1:n_groups[tt_]
                                idx = find(x -> x>kk, zz[tt_][jj_])
                                zz[tt_][jj_][idx] -= 1

                                tbl_idx = unique(tji[tt_][jj_][idx])
                                kjt[tt_][jj_][tbl_idx] -= 1
                            end
                        end # for tt_

                        splice!(components, kk)
                        mm = del_column(mm, kk)
                        rcrf.KK -= 1
                    end
                else
                    for ll in tidx
                        delitem!(components[kk], xx[tt][jj][ll])
                    end
                end



                kk_idx_precur = find(x -> x>0, sum(mm[tt-1:tt, :], 1))

                # resample the dish
                pp = fill(-Inf, rcrf.KK+1)
                for kk in kk_idx_precur
                    pp[kk] = log(mm[tt-1, kk] + mm[tt, kk])
                    for ll in tidx
                        pp[kk] += logpredictive(components[kk], xx[tt][jj][ll])
                    end
                end
                pp[rcrf.KK+1] = log(rcrf.gg[tt])
                for ll in tidx
                    pp[rcrf.KK+1] += logpredictive(rcrf.component, xx[tt][jj][ll])
                end


                lognormalize!(pp)
                kk = sample(pp)


                if kk == rcrf.KK+1
                    push!(components, deepcopy(rcrf.component))
                    rcrf.KK += 1
                    mm = add_column(mm)
                end

                kjt[tt][jj][tbl] = kk
                for ll in tidx
                    additem!(components[kk], xx[tt][jj][ll])
                end
                mm[tt, kk] += 1
                zz[tt][jj][tidx] = kk
            end # for tbl
        end # for jj

        # resample hyper-parameters
        if sample_hyperparam
            KK_tt = length(findnz(mm[tt, :])[1])
            m = sum([length(kjt[tt][jj]) for jj=1:n_groups[tt]])
            sample_hyperparam!(rcrf, tt, n_group_j[tt], m, KK_tt, n_internals)
        end

        elapsed_time = toq()

        # save the sample
        if (iteration-n_burnins) % (n_lags+1) == 0 && iteration > n_burnins
            sample_n = n_samples_old + convert(Int, (iteration-n_burnins)/(n_lags+1))
            KK_list[sample_n] = rcrf.KK
            # rcrf_sample = RCRFSample(deepcopy(njt), deepcopy(kjt), deepcopy(zz))
            # KK_dict[rcrf.KK] = rcrf_sample
            KK_dict[rcrf.KK] = deepcopy(zz)
            if (sample_n % store_every) == 0
                storesample(rcrf, KK_list, KK_dict, n_burnins, n_lags, sample_n, filename)
            end
        end

        log_likelihood = 0.0
        for tt = 1:TT
            for jj = 1:n_groups[tt]
                for ii = 1:n_group_j[tt][jj]
                    log_likelihood += loglikelihood(components[zz[tt][jj][ii]], xx[tt][jj][ii])
                end
            end
        end


    end # iteration

    sample_n = n_samples_old +  convert(Int, (n_iterations-n_burnins)/(n_lags+1))
    KK_list[sample_n] = rcrf.KK
    # rcrf_sample = RCRFSample(deepcopy(njt), deepcopy(kjt), deepcopy(zz))
    # _dict[rcrf.KK] = rcrf_sample
    KK_dict[rcrf.KK] = deepcopy(zz)
    storesample(rcrf, KK_list, KK_dict, n_burnins, n_lags, sample_n, filename)

    KK_list, KK_dict

end # RCRF_gibbs_sampler!()



function posterior{T1, T2}(
    rcrf::RCRF{T1},
    xx::Vector{Vector{Vector{T2}}},
    KK_dict::Dict{Int, Vector{Vector{Vector{Int}}}},
    KK::Int)


    TT        = length(xx)
    n_groups  = [length(xx[tt]) for tt=1:TT]
    n_group_j = Array(Vector{Int}, TT)
    nn        = Array(Matrix{Int}, TT)
    zz        = KK_dict[KK]


    for tt = 1:TT
        n_group_j[tt] = zeros(Int, n_groups[tt])
        for jj = 1:n_groups[tt]
            n_group_j[tt][jj] = length(xx[tt][jj])
        end
    end


    components = Array(typeof(rcrf.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(rcrf.component)
    end


    for tt = 1:TT
        nn[tt] = zeros(Int, n_groups[tt], KK)
        for jj = 1:n_groups[tt]
            for ii = 1:n_group_j[tt][jj]
                kk = zz[tt][jj][ii]
                nn[tt][jj, kk] += 1
                additem!(components[kk], xx[tt][jj][ii])
            end
        end
    end

    pos_components = Array(typeof(posterior(rcrf.component)), KK)
    for kk = 1:KK
        pos_components[kk] = posterior(components[kk])
    end

    pos_components, KK_dict[KK], nn
end



function posterior{T1, T2}(
    rcrf::RCRF{T1},
    xx::Vector{Vector{Vector{T2}}},
    KK_dict::Dict{Int, RCRFSample},
    KK::Int)


    TT        = length(xx)
    n_groups  = [length(xx[tt]) for tt=1:TT]
    n_group_j = Array(Vector{Int}, TT)
    nn        = Array(Matrix{Int}, TT)
    mm        = zeros(Int, TT, KK)
    zz        = KK_dict[KK].zz


    for tt = 1:TT
        n_group_j[tt] = zeros(Int, n_groups[tt])
        for jj = 1:n_groups[tt]
            n_group_j[tt][jj] = length(xx[tt][jj])
        end
    end


    components = Array(typeof(rcrf.component), KK)
    for kk = 1:KK
        components[kk] = deepcopy(rcrf.component)
    end

    for tt = 1:TT
        for jj = 1:n_groups[tt]
            for kk in KK_dict[KK].kjt[tt][jj]
                mm[tt, kk] += 1
            end
        end
    end


    for tt = 1:TT
        nn[tt] = zeros(Int, n_groups[tt], KK)
        for jj = 1:n_groups[tt]
            for ii = 1:n_group_j[tt][jj]
                kk = zz[tt][jj][ii]
                nn[tt][jj, kk] += 1
                additem!(components[kk], xx[tt][jj][ii])
            end
        end
    end

    pos_components = Array(typeof(posterior(rcrf.component)), KK)
    for kk = 1:KK
        pos_components[kk] = posterior(components[kk])
    end

    pos_components, KK_dict[KK].njt, KK_dict[KK].kjt, nn, mm
end
