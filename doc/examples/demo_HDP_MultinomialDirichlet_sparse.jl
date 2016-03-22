#=
demo_HDP_MultinomialDirichlet_sparse.jl

A demo for Hierarchical Dirichlet Process mixture models with MultinomialDirichlet 
Bayesian components. This demo uses auxiliary variable method for for inference.

15/09/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS

## --- synthesizing the data --- ##
# synthesized corpus properties
true_KK    = 10
n_groups   = 100
n_group_j  = 50 * ones(Int64, n_groups)
n_tokens   = 25
vocab_size = 25

# constructing the topics
true_topics = BIAS.gen_bars(true_KK, vocab_size, 0.02)

# constructing the observations and labels
alpha = 0.1
xx = Array(Vector{Sent}, n_groups)
true_zz = Array(Vector{Int64}, n_groups)
true_nn = zeros(n_groups, true_KK)
for jj = 1:n_groups
    xx[jj] = Array(Sent, n_group_j[jj])
    true_zz[jj] = ones(Int64, n_group_j[jj])
    theta = BIAS.rand_Dirichlet(fill(alpha, true_KK))
    for ii = 1:n_group_j[jj]
        kk = BIAS.sample(theta)
        sentence = BIAS.sample(true_topics[kk, :][:], n_tokens)
        xx[jj][ii] = BIAS.sparsify_sentence(sentence)
        true_zz[jj][ii] = kk
        true_nn[jj, kk] += 1
    end
end


## ------- inference --------
# constructing the Bayesian component
dd = vocab_size
aa = 0.01*dd
q0 = MultinomialDirichlet(dd, aa)
# constructing the model
KK_init = 1
hdp_gg = 1
hdp_g1 = 0.1
hdp_g2 = 0.1
hdp_aa = 0.5
hdp_a1 = 0.1
hdp_a2 = 0.1
hdp = HDP(q0, KK_init, hdp_gg, hdp_g1, hdp_g2, hdp_aa, hdp_a1, hdp_a2)

# sampling
zz = Array(Vector{Int64}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int64, n_group_j[jj])
end

init_zz!(hdp, zz)

n_burnins   = 50
n_lags      = 1
n_samples   = 50
sample_hyperparam = true
n_internals = 10
store_every = 100
filename    = "demo_HDP_MultinomialDirichlet_"
K_list, K_zz_dict, betas, gammas, alphas = collapsed_gibbs_sampler!(hdp, xx, zz,
        n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)

# posterior distributions
K_hist = hist(K_list, 0.5:maximum(K_list)+0.5)[2]
candidate_K = indmax(K_hist)
posterior_components, nn = posterior(hdp, xx, K_zz_dict, candidate_K)

# constructing the inferred topics
inferred_topics = zeros(Float64, length(posterior_components), vocab_size)
for kk = 1:length(posterior_components)
    inferred_topics[kk, :] = mean(posterior_components[kk])
end

# visualizing the results
visualize_bartopics(true_topics)
visualize_bartopics(inferred_topics)
