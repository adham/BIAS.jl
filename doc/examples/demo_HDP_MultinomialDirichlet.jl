#=
LDA_demo_MultinomialDirichlet

A demo for LDA with MultinomialDIrichlet Bayesian components.

28/07/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS
srand(123)

## --- synthesizing the data --- ##

# synthesized corpus properties
true_KK    = 5
n_groups   = 50
n_group_j  = 100 * ones(Int, n_groups)
vocab_size = 25
max_n_topic_doc = 3

# constructing the topics
true_topics = BIAS.gen_bars(true_KK, vocab_size, 0)

# constructing the observations and labels
xx = Array(Vector{Int}, n_groups)
true_zz = Array(Vector{Int}, n_groups)
true_nn = zeros(Int, n_groups, true_KK)
n_topic_doc_p = cumsum(fill(1/max_n_topic_doc, max_n_topic_doc))

for jj = 1:n_groups
    n_topic_doc = sum(rand() .< n_topic_doc_p)
    topic_doc = randperm(true_KK)[1:n_topic_doc]

    xx[jj] = zeros(Int, n_group_j[jj])
    true_zz[jj] = ones(Int, n_group_j[jj])
    for ii = 1:n_group_j[jj]
        kk = topic_doc[rand(1:n_topic_doc)]
        true_zz[jj][ii] = kk
        xx[jj][ii] = BIAS.sample(true_topics[kk, :][:])
        true_nn[jj, kk] += 1
    end
end

## ------- inference -------- ##
# constructing the Bayesian component
dd = vocab_size
aa = 0.5*dd
q0 = MultinomialDirichlet(dd, aa)

# constructing the  model
KK_init = 1
hdp_gg = 0.1
hdp_g1 = 1
hdp_g2 = 0.1
hdp_aa = 1
hdp_a1 = 1
hdp_a2 = 1
hdp = HDP(q0, KK_init, hdp_gg, hdp_g1, hdp_g2, hdp_aa, hdp_a1, hdp_a2)

# sampling
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int, n_group_j[jj])
end

init_zz!(hdp, zz)

n_burnins   = 100
n_lags      = 1
n_samples   = 200
sample_hyperparam = true
n_internals = 20
store_every = 10000
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

visualize_bartopics(inferred_topics)
