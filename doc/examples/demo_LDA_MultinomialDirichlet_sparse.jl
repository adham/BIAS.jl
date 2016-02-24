#=
demo_LDA_MultinomialDirichlet_sparse

Demo for Latent Dirichlet Allocation mixture model with MultinomialDirichlet conjugates with
multiple observations i.e. each draw is a sentence.

28/07/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS
srand(123)


## --- synthesizing the data --- ##
true_KK = 10
n_groups = 25
n_group_j = 100 * ones(Int, n_groups)
n_tokens = 20
vocab_size = 25

# constructing the topics
true_topics = BIAS.gen_bars(true_KK, vocab_size, 0)

# constructing the observations and labels
alpha = 0.1
xx = Array(Vector{Sent}, n_groups)
true_zz = Array(Vector{Int}, n_groups)
true_nn = zeros(Int, n_groups, true_KK)
for jj = 1:n_groups
    xx[jj] = Array(Sent, n_group_j[jj])
    true_zz[jj] = ones(Int, n_group_j[jj])
    theta = BIAS.rand_Dirichlet(fill(alpha, true_KK))
    for ii = 1:n_group_j[jj]
        kk = sample(theta)
    sentence = sample(true_topics[kk, :][:], n_tokens)
    xx[jj][ii] = BIAS.sparsify_sentence(sentence)
    true_zz[jj][ii] = kk
    true_nn[jj, kk] += 1
  end
end


## ------- inference -------- ##
# constructing the Bayesian component
dd = vocab_size
aa = 0.1 * dd
q0 = MultinomialDirichlet(dd, aa)

# constructing the model
lda_KK = true_KK
lda_aa = 0.1
lda = LDA(q0, lda_KK, lda_aa)

# sampling
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int, n_group_j[jj])
end
init_zz!(lda, zz)

n_burnins   = 100
n_lags      = 2
n_samples   = 200
store_every = 100
filename    = "demo_LDA_Multinomial_Dirichlet"

collapsed_gibbs_sampler!(lda, xx, zz, n_burnins, n_lags, n_samples, store_every, filename)


# posterior distributions
posterior_components, nn = posterior(lda, xx, zz)

# constructing the inferred topics
inferred_topics = zeros(Float64, length(posterior_components), vocab_size)
for kk = 1:length(posterior_components)
    inferred_topics[kk, :] = mean(posterior_components[kk])
end

# visualizing the results
visualize_bartopics(inferred_topics)
