#=
demo_RCRF_MultinomialDirichlet.jl

demo for Recurrent Chinese Restaurant Franchise with MultinomialDirichlet
conjugate components.

22/03/2016
Adham Beyki, odinay@gmail.com
=#

using BIAS

## --- synthesizing the data ---
TT = 2
n_groups = fill(100, TT)
n_group_j = Array(Vector{Int}, TT)
n_tokens   = 1000
vocab_size = 4

for tt = 1:TT
    n_group_j[tt] = fill(n_tokens, n_groups[tt])
end

true_aa = 0.8
true_gg = 0.5
join_tables = true

true_njt, true_kjt, true_nn, true_mm, true_zz, true_KK = BIAS.gen_RCRF_data(n_group_j, true_gg, true_aa, join_tables)


# constructing the topics
true_topics = BIAS.gen_bars(true_KK, vocab_size, 0)

xx = Array(Vector{Vector{Int}}, TT)
for tt = 1:TT
    xx[tt] = Array(Vector{Int}, n_groups[tt])

    for jj = 1:n_groups[tt]
        xx[tt][jj] = zeros(Int, n_group_j[tt][jj])
        for ii = 1:n_group_j[tt][jj]
            kk = true_zz[tt][jj][ii]
            xx[tt][jj][ii] = sample(true_topics[kk, :][:])
        end
    end
end


dd = vocab_size
aa = 1.0
q0 = MultinomialDirichlet(dd, aa)

init_KK = 1
rcrf_gg = true_gg
rcrf_g1 = 0.1
rcrf_g2 = 0.1
rcrf_aa = true_aa
rcrf_a1 = 0.1
rcrf_a2 = 0.1

rcrf = RCRF(q0, init_KK, rcrf_gg, rcrf_g1, rcrf_g2, rcrf_aa, rcrf_a1, rcrf_a2, TT)

zz = Array(Vector{Vector{Int}}, TT)
for tt = 1:TT
    zz[tt] = Array(Vector{Int}, n_groups[tt])
    for jj = 1:n_groups[tt]
        zz[tt][jj] = rand(1:init_KK, n_group_j[tt][jj])
    end
end
#init_zz(rcrf, zz)

n_burnins   = 100
n_lags      = 1
n_samples   = 200
sample_hyperparam = false
n_internals = 10
store_every = 100000
filename    = "demo_RCRF_Gaussian1DGaussian1D_"


KK_list, KK_dict = RCRF_gibbs_sampler!(rcrf, xx, zz, n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename, join_tables)

KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_KK = indmax(KK_hist)


pos_components, njt, kjt, nn, mm = posterior(rcrf, xx, KK_dict, candidate_KK)

inferred_topics = zeros(Float64, length(pos_components), vocab_size)
for kk = 1:length(pos_components)
    inferred_topics[kk, :] = mean(pos_components[kk])
end

visualize_bartopics(true_topics)
visualize_bartopics(inferred_topics)
