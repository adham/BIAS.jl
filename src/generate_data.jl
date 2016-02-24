#=
generate_data.jl

generates data based on the generative process of the following models:

    - CRP
    - CRF
    - RCRP
    - RCRF

TODO:
implement CRP
use multiple dispatch and package all these function under one name
return mm in crf

11/02/2016
Adham Beyki, odinay@gmail.com
=#








function gen_RCRP_data(N_t::Vector{Int}, aa::Float64)

    true_KK = 1
    TT = length(N_t)
    true_nn = zeros(Int, TT, true_KK)
    true_zz = Array(Vector{Int}, TT)

    tt = 1
    true_zz[tt] = zeros(Int, N_t[tt])

    ii = 1
    kk = 1
    true_zz[tt][ii] = kk
    true_nn[tt, kk] += 1

    for ii = 2:N_t[tt]
        pp = zeros(Float64, true_KK+1)
        for kk = 1:true_KK
            pp[kk] = true_nn[tt, kk] / (aa + ii-1)
        end
        pp[true_KK+1] = aa / (aa + ii-1)

        kk = sample(pp)

        if kk == true_KK+1
            true_nn = BIAS.add_column(true_nn)
            true_KK += 1
        end

        true_zz[tt][ii] = kk
        true_nn[tt, kk] += 1
    end

    for tt = 2:TT
        true_zz[tt] = zeros(Int, N_t[tt])

        for ii = 1:N_t[tt]
            pp = zeros(Float64, true_KK+1)
            for kk = 1:true_KK
                pp[kk] = (true_nn[tt-1, kk] + true_nn[tt, kk]) / (N_t[tt-1]+ii-1+aa)
            end
            pp[true_KK+1] = aa / (N_t[tt-1]+ii-1+aa)

            kk = sample(pp)

            if kk == true_KK+1
                true_nn = BIAS.add_column(true_nn)
                true_KK += 1
            end

            true_zz[tt][ii] = kk
            true_nn[tt, kk] += 1
        end
    end

    return true_nn, true_zz
end


function gen_CRF_data(n_group_j::Vector{Int}, gg::Float64, aa::Float64)


    n_groups = length(n_group_j)
    zz = Array(Vector{Int}, n_groups)
    njt = Array(Vector{Int}, n_groups)
    kjt = Array(Vector{Int}, n_groups)

    KK = 1
    mm = zeros(Int, KK)
    nn = zeros(Int, n_groups, KK)


    jj = 1
    TT = 1
    zz[jj] = zeros(Int, n_group_j[jj])
    njt[jj] = zeros(Int, TT)
    kjt[jj] = zeros(Int, TT)


    ii = 1
    tt = 1
    kk = 1

    njt[jj][tt] += 1
    kjt[jj][tt] = kk
    zz[jj][ii] = kk
    nn[jj, kk] += 1
    mm[kk] += 1


    for ii = 2:n_group_j[jj]
        pp = zeros(Float64, TT+1)
        for tt=1:TT
            pp[tt] = njt[jj][tt] / (aa + ii-1)
        end
        pp[TT+1] = aa / (aa + ii-1)

        tt = sample(pp)
        if tt == TT+1
            TT += 1
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
            kjt[jj][tt] = kk
            mm[kk] += 1
        end

        njt[jj][tt] += 1
        kk = kjt[jj][tt]
        zz[jj][ii] = kk
        nn[jj, kk] += 1    
        
    end
    

    for jj = 2:n_groups
        
        # no one is in yet. There is only one table.
        TT = 1
        
        zz[jj] = zeros(Int, n_group_j[jj])
        njt[jj] = zeros(Int, TT)
        kjt[jj] = zeros(Int, TT)
        
        # customer #1 sits at table #1
        ii = 1
        tt = 1    
        njt[jj][tt] += 1
        
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

        kjt[jj][tt] = kk
        mm[kk] += 1
        zz[jj][ii] = kk
        nn[jj, kk] += 1

       

        for ii = 2:n_group_j[jj]


            # sample table
            pp = zeros(Float64, TT+1)
            for tt=1:TT
                pp[tt] = njt[jj][tt] / (aa + ii-1)
            end
            pp[TT+1] = aa / (aa + ii-1)

            tt = sample(pp)
            if tt == TT+1       # if new table is instantiated

                TT += 1
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
                kjt[jj][tt] = kk
                mm[kk] += 1
            end

            njt[jj][tt] += 1
            kk = kjt[jj][tt]
            zz[jj][ii] = kk
            nn[jj, kk] += 1
        end
    end

    return(njt, kjt, nn, zz, KK)
end


function gen_CRP_data(NN::Int, aa::Float64)
    
    KK = 1
    
    zz  = zeros(Int, NN)
    nn = zeros(Int, KK)
    
    ii = 1
    kk = 1
    
    zz[ii]  = kk
    nn[kk] += 1
    
    for ii = 2:NN
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