#=
demo_LDA_MultinomialDirichlet.jl

Demo for Latent Dirichlet Allocation mixture model with MultinomialDirichlet conjugates with
single observations i.e. each draw is a word.

28/07/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS

## --- synthesizing the data --- ##
true_KK = 10
n_groups = 100
n_group_j = 50 * ones(Int, n_groups)
vocab_size = 25
max_n_topic_doc = 4

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
        xx[jj][ii] = sample(true_topics[kk, :][:])
        true_nn[jj, kk] += 1
    end
end

## ------- inference -------- ##
# constructing the conjugate
dd = vocab_size
aa = 0.1 * dd
q0 = MultinomialDirichlet(dd, aa)

# constructing the LDA model
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
n_samples   = 400
store_every = 100
filename    = "demo_LDA_MultinomialDirichlet_"

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
