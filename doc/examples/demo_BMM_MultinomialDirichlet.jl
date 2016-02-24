#=
demo_BMM_MultinomialDirichlet.jl

Demo for Bayesian Finite Mixture Model with MultinomialDirichlet conjugates.

08/07/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS
srand(123)

## --- synthesizing the data ---
true_KK     = 4
n_sentences = 200
n_tokens    = 15
vocab_size  = 25

true_topics = BIAS.gen_bars(true_KK, vocab_size, 0.0)

# constructing the observations and labels
mix = ones(true_KK) / true_KK
xx = Array(Sent, n_sentences)
true_zz = zeros(Int, n_sentences)
true_nn = zeros(Int, true_KK)
for ii = 1:n_sentences
    kk = sample(mix)
    true_zz[ii] = kk
    true_nn[kk] += 1
    sentence = sample(true_topics[kk, :][:], n_tokens)
    xx[ii] = BIAS.sparsify_sentence(sentence)
end


## ------- inference -------- ##
# constructing the Bayesian component
dd = vocab_size
aa = 1.0
q0 = MultinomialDirichlet(dd, aa)

# constructing the model
bmm_KK = true_KK
bmm_aa = 0.1
bmm = BMM(q0, bmm_KK, bmm_aa)

# Sampling
zz = zeros(Int, length(xx))
init_zz!(bmm, zz)

n_burnins   = 100
n_lags      = 0
n_samples   = 200
store_every = 1000
filename    = "demo_BMM_MultinomialDirichlet_"

collapsed_gibbs_sampler!(bmm, xx, zz, n_burnins, n_lags, n_samples, store_every, filename)

# posterior distributions
posterior_components, nn = posterior(bmm, xx, zz)

# constructing the inferred topics
inferred_topics = zeros(Float64, bmm.KK, vocab_size)
for kk = 1:length(posterior_components)
    inferred_topics[kk, :] = mean(posterior_components[kk])
end

# visualizing the results
visualize_bartopics(inferred_topics)
