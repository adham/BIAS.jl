#=
demo_RCRP_MultinomialDirichlet_sparse

demo for Recurrent Chinese Restaurant Process with MultinomialDIrichlet conjugate component.

15/01/2015
Adham Beyki, odinay@gmail.com
=#


using BIAS

srand(100)

## --- synthesizing the data --- ##

TT      = 10
N_t     = fill(100, TT)
true_aa = 0.5

true_nn, true_zz, true_KK = BIAS.gen_RCRP_data(N_t, true_aa)

true_KK    = size(true_nn, 2)
n_tokens   = 25
vocab_size = 36

# constructing the topics
true_topics = BIAS.gen_bars(true_KK, vocab_size, 0)

# constructing the observations
xx = Array(Vector{Sent}, TT)
for tt = 1:TT
    xx[tt] = Array(Sent, N_t[tt])
    for ii = 1:N_t[tt]
        kk = true_zz[tt][ii]
        sentence = sample(true_topics[kk, :][:], n_tokens)
        xx[tt][ii] = BIAS.sparsify_sentence(sentence)
    end
end



## ------- inference -------- ##

# constructing the conjugate
dd = vocab_size
aa = 1.0
q0 = MultinomialDirichlet(dd, aa)

# constructing the model
init_KK = 1
rcrp_aa = 0.5
rcrp_a1 = 1
rcrp_a2 = 1
rcrp = RCRP(q0, init_KK, rcrp_aa, TT, rcrp_a1, rcrp_a2)


# sampling
zz = Array(Vector{Int64}, TT)
for tt = 1:TT
  zz[tt] = rand(1:rcrp.KK, N_t[tt])
end

n_burnins   = 10
n_lags      = 0
n_samples   = 10
sample_hyperparam = true
n_internals = 10
store_every = 10000
filename    = "demo_RCRP_MultinomialDirichlet_"
KK_list, KK_dict = RCRP_gibbs_sampler!(rcrp, xx, zz,
    n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)

KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_KK = indmax(KK_hist)

pos_components, nn = posterior(rcrp, xx, KK_dict, candidate_KK)


# posterior distributions
inferred_topics = zeros(Float64, length(pos_components), vocab_size)
for k_t = 1:length(pos_components)
    inferred_topics[k_t, :] = mean(pos_components[k_t])
end

visualize_bartopics(true_topics)
visualize_bartopics(inferred_topics)