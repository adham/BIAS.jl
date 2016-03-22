#=
generate_data.jl

generates data based on the generative process of the following models:

    - CRP
    - CRF
    - RCRP
    - RCRF

TODO:
    use multiple dispatch and package all these function under one name
    
    RCRP: aa could be a vector. Each epoch different concentraion parameter
    RCRF: aa and gg could be vectors


11/02/2016
Adham Beyki, odinay@gmail.com
=#


function gen_CRP_data(n_customers::Int, aa::Float64)

    KK = 1

    zz  = zeros(Int, n_customers)
    nn = zeros(Int, KK)

    ii = 1
    kk = 1

    zz[ii]  = kk
    nn[kk] += 1

    for ii = 2:n_customers
        pp = zeros(Float64, KK+1)
        for kk = 1:KK
            pp[kk] = nn[kk] / (aa + ii-1)
        end
        pp[KK+1] = aa / (aa + ii-1)

        kk = sample(pp)

        if kk == KK+1
            push!(nn, 0)
            KK += 1
        end
        nn[kk] += 1
        zz[ii] = kk
    end

    nn, zz, KK
end


function gen_RCRP_data(N_t::Vector{Int}, aa::Float64)

    KK = 1
    TT = length(N_t)
    nn = zeros(Int, TT, KK)
    zz = Array(Vector{Int}, TT)

    tt = 1
    zz[tt] = zeros(Int, N_t[tt])

    ii = 1
    kk = 1
    zz[tt][ii] = kk
    nn[tt, kk] += 1

    for ii = 2:N_t[tt]
        pp = zeros(Float64, KK+1)
        for kk = 1:KK
            pp[kk] = nn[tt, kk] / (aa + ii-1)
        end
        pp[KK+1] = aa / (aa + ii-1)

        kk = sample(pp)

        if kk == KK+1
            nn = BIAS.add_column(nn)
            KK += 1
        end

        zz[tt][ii] = kk
        nn[tt, kk] += 1
    end

    for tt = 2:TT
        zz[tt] = zeros(Int, N_t[tt])

        for ii = 1:N_t[tt]
            pp = zeros(Float64, KK+1)
            for kk = 1:KK
                pp[kk] = (nn[tt-1, kk] + nn[tt, kk]) / (N_t[tt-1]+ii-1+aa)
            end
            pp[KK+1] = aa / (N_t[tt-1]+ii-1+aa)

            kk = sample(pp)

            if kk == KK+1
                nn = BIAS.add_column(nn)
                KK += 1
            end

            zz[tt][ii] = kk
            nn[tt, kk] += 1
        end
    end

    return nn, zz, KK
end


function gen_CRF_data(n_group_j::Vector{Int}, gg::Float64, aa::Float64, join_tables::Bool)

    n_groups = length(n_group_j)
    zz       = Array(Vector{Int}, n_groups)
    tji      = Array(Vector{Int}, n_groups)
    njt      = Array(Vector{Int}, n_groups)
    kjt      = Array(Vector{Int}, n_groups)

    KK = 1
    mm = zeros(Int, KK)
    nn = zeros(Int, n_groups, KK)


    jj      = 1
    n_tbls  = 1
    zz[jj]  = zeros(Int, n_group_j[jj])
    tji[jj] = zeros(Int, n_group_j[jj])
    njt[jj] = zeros(Int, n_tbls)
    kjt[jj] = zeros(Int, n_tbls)


    ii  = 1
    tbl = 1
    kk  = 1
    kjt[jj][tbl] = kk
    tji[jj][ii]  = tbl
    njt[jj][tbl] += 1
    zz[jj][ii]   = kjt[jj][tbl]
    nn[jj, kk]   += 1
    mm[kk]       += 1


    for ii = 2:n_group_j[jj]
        pp = zeros(Float64, n_tbls+1)
        for tbl=1:n_tbls
            pp[tbl] = njt[jj][tbl] / (aa + ii-1)
        end
        pp[n_tbls+1] = aa / (aa + ii-1)

        tbl = sample(pp)
        if tbl == n_tbls+1
            n_tbls += 1
            push!(njt[jj], 0)
            push!(kjt[jj], 0)

            pp = zeros(Float64, KK+1)
            for kk = 1:KK
                pp[kk] = mm[kk] / (gg + sum(mm))
            end
            pp[KK+1] = gg / (gg + sum(mm))

            kk = sample(pp)

            if kk == KK+1
                push!(mm, 0)
                nn = add_column(nn)
                KK += 1
            end
            kjt[jj][tbl] = kk
            mm[kk] += 1
        end

        tji[jj][ii] = tbl
        njt[jj][tbl] += 1
        kk = kjt[jj][tbl]
        zz[jj][ii] = kk
        nn[jj, kk] += 1

    end


    for jj = 2:n_groups

        # no one is in yet. There is only one table.
        n_tbls = 1

        zz[jj]  = zeros(Int, n_group_j[jj])
        tji[jj] = zeros(Int, n_group_j[jj])
        njt[jj] = zeros(Int, n_tbls)
        kjt[jj] = zeros(Int, n_tbls)

        # customer #1 sits at table #1
        ii = 1
        tbl = 1
        tji[jj][ii] = tbl
        njt[jj][tbl] += 1

        # sample the dish for table #1
        pp = zeros(Float64, KK+1)
        for kk = 1:KK
            pp[kk] = mm[kk] / (gg + sum(mm))
        end
        pp[KK+1] = gg / (gg + sum(mm))

        kk = sample(pp)
        if kk == KK+1
            push!(mm, 0)
            nn = add_column(nn)
            KK += 1
        end

        tji[jj][ii]  = tbl
        kjt[jj][tbl] = kk
        mm[kk] += 1
        zz[jj][ii] = kk
        nn[jj, kk] += 1


        for ii = 2:n_group_j[jj]

            # sample table
            pp = zeros(Float64, n_tbls+1)
            for tbl=1:n_tbls
                pp[tbl] = njt[jj][tbl] / (aa + ii-1)
            end
            pp[n_tbls+1] = aa / (aa + ii-1)

            tbl = sample(pp)
            if tbl == n_tbls+1       # if new table is instantiated

                n_tbls += 1
                push!(njt[jj], 0)
                push!(kjt[jj], 0)

                # sample the dish for the newly initiated table
                pp = zeros(Float64, KK+1)
                for kk = 1:KK
                    pp[kk] = mm[kk] / (gg + sum(mm))
                end
                pp[KK+1] = gg / (gg + sum(mm))

                kk = sample(pp)

                # if it's a new dish
                if kk == KK+1
                    push!(mm, 0)
                    nn = add_column(nn)
                    KK += 1
                end
                kjt[jj][tbl] = kk
                mm[kk] += 1
            end

            tji[jj][ii] = tbl
            njt[jj][tbl] += 1
            kk = kjt[jj][tbl]
            zz[jj][ii] = kk
            nn[jj, kk] += 1
        end
    end


    # joining the tables that serve the same dish in each restaurant
    if join_tables
        for jj = 1:n_groups
            kk_list = unique(kjt[jj])
            new_kjt = zeros(Int, length(kk_list))
            new_njt = zeros(Int, length(kk_list))

            ll = 1
            for unq_kk in kk_list
                idx = find(x -> x==unq_kk, kjt[jj])
                new_kjt[ll] = unq_kk
                new_njt[ll] = sum(njt[jj][idx])
                ll += 1
            end

            kjt[jj] = new_kjt
            njt[jj] = new_njt
        end

        mm = zeros(Int, KK)
        for jj = 1:n_groups
            for kk in kjt[jj]
                mm[kk] += 1
            end
        end
    end

    return(tji, njt, kjt, nn, mm, zz, KK)
end



function gen_RCRF_data(n_group_j::Vector{Vector{Int}}, gg::Float64, aa::Float64, join_tables::Bool=false)


    TT       = length(n_group_j)                        # number of epochs
    n_groups = [length(foo) for foo in n_group_j]       # number of groups per epoch


    # initial KK
    KK = 1

    zz  = Array(Vector{Vector{Int}}, TT)                # zz[tt][jj][ii] is the cluster customer ii at restaurant jj in epoch tt belongs to
    njt = Array(Vector{Vector{Int}}, TT)                # njt[tt][jj][tbl] is the number of customers sitting at table tbl at restaurant jj in epoch tt
    kjt = Array(Vector{Vector{Int}}, TT)                # kjt[tt][jj][tbl] is the dish being served at table tbl at restaurant jj in epoch tt
    mm  = zeros(Int, TT, KK)                            # mm[tt, kk] is the number of times dish kk served at a table in epoch tt
    nn  = Array(Matrix{Int}, TT)                        # nn[tt][jj, kk] is the number of observations that belong to cluster kk in restaurant jj at epoch tt


    # we start with epoch 1, restaurant 1, customer 1
    # Initializing epoch 1
    tt      = 1
    zz[tt]  = Array(Vector{Int}, n_groups[tt])
    njt[tt] = Array(Vector{Int}, n_groups[tt])
    kjt[tt] = Array(Vector{Int}, n_groups[tt])
    nn[tt]  = zeros(Int, n_groups[tt], KK)

    # Initializing restaurant 1
    jj          = 1
    n_tbls      = 1
    zz[tt][jj]  = zeros(Int, n_group_j[tt][jj])
    njt[tt][jj] = zeros(Int, n_tbls)
    kjt[tt][jj] = zeros(Int, n_tbls)


    # customer 1 enters restaurant 1 at epoch 1, picks the the first table
    # and orders the first dish
    ii  = 1
    kk  = 1
    tbl = 1

    njt[tt][jj][tbl] += 1
    kjt[tt][jj][tbl] = kk
    zz[tt][jj][ii]   = kk
    mm[tt, kk]       += 1
    nn[tt][jj, kk]   += 1


    # table assignment for customers 2:end in restaurant 1 epoch 1
    for ii = 2:n_group_j[tt][jj]

        pp = zeros(Float64, n_tbls+1)
        for tbl = 1:n_tbls
            pp[tbl] = njt[tt][jj][tbl] / (aa + ii-1)
        end
        pp[n_tbls+1] = aa / (aa + ii-1)

        tbl = sample(pp)

        # a new table is instantiated
        if tbl == n_tbls+1

            n_tbls += 1
            push!(njt[tt][jj], 0)
            push!(kjt[tt][jj], 0)

            # sample the dish for the newly instantiated table
            pp = zeros(Float64, KK+1)
            for kk = 1:KK
                pp[kk] = mm[tt, kk] / (gg + sum(mm[tt, :]))
            end
            pp[KK+1] = gg / (gg + sum(mm[tt, :]))

            kk = sample(pp)

            # if it is a new dish
            if kk == KK+1
                mm = add_column(mm)
                nn[tt] = add_column(nn[tt])
                KK += 1
            end
            kjt[tt][jj][tbl] = kk
            mm[tt, kk] += 1
        end

        njt[tt][jj][tbl] += 1
        kk = kjt[tt][jj][tbl]
        zz[tt][jj][ii] = kk
        nn[tt][jj, kk] += 1

    end

    for jj = 2:n_groups[tt]

        # no one is in yet
        n_tbls = 1

        zz[tt][jj]  = zeros(Int, n_group_j[tt][jj])
        njt[tt][jj] = zeros(Int, n_tbls)
        kjt[tt][jj] = zeros(Int, n_tbls)

        # customer 1 sits at the first table
        ii = 1
        tbl = 1
        njt[tt][jj][tbl] += 1

        # sample the dish for table 1
        pp = zeros(Float64, KK+1)
        for kk = 1:KK
            pp[kk] = mm[tt, kk] / (gg + sum(mm[tt, :]))
        end
        pp[KK+1] = gg / (gg + sum(mm[tt, :]))

        kk = sample(pp)
        if kk == KK+1
            mm = add_column(mm)
            nn[tt] = add_column(nn[tt])
            KK += 1
        end

        kjt[tt][jj][tbl] = kk
        zz[tt][jj][ii]   = kk
        mm[tt, kk]       += 1
        nn[tt][jj, kk]   += 1

        for ii = 2:n_group_j[tt][jj]

            # sample the table for customer ii
            pp = zeros(Float64, n_tbls+1)
            for tbl = 1:n_tbls
                pp[tbl] = njt[tt][jj][tbl] / (aa + ii-1)
            end
            pp[n_tbls+1] = aa / (aa + ii-1)

            tbl = sample(pp)

            # a new table is instantiated
            if tbl == n_tbls + 1

                n_tbls += 1
                push!(njt[tt][jj], 0)
                push!(kjt[tt][jj], 0)

                # sample the dish for the newly instantiated table
                pp = zeros(Float64, KK+1)
                for kk = 1:KK
                    pp[kk] = mm[tt, kk] / (gg + sum(mm[tt, :]))
                end
                pp[KK+1] = gg / (gg + sum(mm[tt, :]))

                kk = sample(pp)

                # if it is a new dish
                if kk == KK+1
                    mm = add_column(mm)
                    nn[tt] = add_column(nn[tt])
                    KK += 1
                end
                kjt[tt][jj][tbl] = kk
                mm[tt, kk] += 1
            end

            njt[tt][jj][tbl] += 1
            kk = kjt[tt][jj][tbl]
            zz[tt][jj][ii] = kk
            nn[tt][jj, kk] += 1

        end
    end


    # epoch 2:TT
    for tt = 2:TT

        zz[tt]  = Array(Vector{Int}, n_groups[tt])
        njt[tt] = Array(Vector{Int}, n_groups[tt])
        kjt[tt] = Array(Vector{Int}, n_groups[tt])
        nn[tt]  = zeros(Int, n_groups[tt], KK)

        for jj = 1:n_groups[tt]

            n_tbls = 1

            zz[tt][jj]  = zeros(Int, n_group_j[tt][jj])
            njt[tt][jj] = zeros(Int, n_tbls)
            kjt[tt][jj] = zeros(Int, n_tbls)

            # customer 1 sits at the first table
            ii = 1
            tbl = 1
            njt[tt][jj][tbl] += 1

            # sample the dish for table 1
            pp = zeros(Float64, KK+1)
            for kk = 1:KK
                pp[kk] = (mm[tt-1, kk] + mm[tt, kk]) / (gg + sum(mm[tt-1:tt, :]))
            end
            pp[KK+1] = gg / (gg + sum(mm[tt-1:tt, :]))

            kk = sample(pp)
            if kk == KK+1
                mm = add_column(mm)
                nn[tt] = add_column(nn[tt])
                KK += 1
            end

            kjt[tt][jj][tbl] = kk
            zz[tt][jj][ii]   = kk
            mm[tt, kk]       += 1
            nn[tt][jj, kk]   += 1

            for ii = 2:n_group_j[tt][jj]

                # sample the table for customer ii
                pp = zeros(Float64, n_tbls+1)
                for tbl = 1:n_tbls
                    pp[tbl] = njt[tt][jj][tbl] / (aa + ii-1)
                end
                pp[n_tbls+1] = aa / (aa + ii-1)

                tbl = sample(pp)

                # a new table is instantiated
                if tbl == n_tbls + 1

                    n_tbls += 1
                    push!(njt[tt][jj], 0)
                    push!(kjt[tt][jj], 0)

                    # sample the dish for the newly instantiated table
                    pp = zeros(Float64, KK+1)
                    for kk = 1:KK
                        pp[kk] = (mm[tt-1, kk] + mm[tt, kk]) / (gg + sum(mm[tt-1:tt, :]))
                    end
                    pp[KK+1] = gg / (gg + sum(mm[tt-1:tt, :]))

                    kk = sample(pp)

                    # if it is a new dish
                    if kk == KK+1
                        mm = add_column(mm)
                        nn[tt] = add_column(nn[tt])
                        KK += 1
                    end
                    kjt[tt][jj][tbl] = kk
                    mm[tt, kk] += 1
                end

                njt[tt][jj][tbl] += 1
                kk = kjt[tt][jj][tbl]
                zz[tt][jj][ii] = kk
                nn[tt][jj, kk] += 1

            end
        end
    end

    # joining the tables that serve the same dish in each restaurant
    if join_tables

        for tt = 1:TT
            for jj = 1:n_groups[tt]
                kk_list = unique(kjt[tt][jj])
                new_kjt = zeros(Int, length(kk_list))
                new_njt = zeros(Int, length(kk_list))

                ll = 1
                for unq_kk in kk_list
                    idx = find(x -> x==unq_kk, kjt[tt][jj])
                    new_kjt[ll] = unq_kk
                    new_njt[ll] = sum(njt[tt][jj][idx])
                    ll += 1
                end

                kjt[tt][jj] = new_kjt
                njt[tt][jj] = new_njt
            end
        end

        mm = zeros(Int, TT, KK)
        for tt = 1:TT
            for jj = 1:n_groups[tt]
                for kk in kjt[tt][jj]
                    mm[tt, kk] += 1
                end
            end
        end

    end # join_tables

    return(njt, kjt, nn, mm, zz, KK)
end # gen_rcrf_data
