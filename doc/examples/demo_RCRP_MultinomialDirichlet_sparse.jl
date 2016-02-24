#=
demo_RCRP_MultinomialDirichlet_sparse

demo for Recurrent Chinese Restaurant Process with
MultinomialDIrichlet conjugate component.

15/01/2015
Adham Beyki, odinay@gmail.com
=#


using BIAS

srand(100)

## --- synthesizing the data --- ##

TT      = 10
N_t     = fill(500, TT)
true_aa = 0.8

true_nn, true_zz = BIAS.gen_RCRP_data(N_t, true_aa)

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
  zz[tt] = rand(1:rcrp.K, N_t[tt])
end

n_burnins   = 10
n_lags      = 0
n_samples   = 10
sample_hyperparam = true
n_internals = 10
store_every = 10000
filename    = "demo_RCRP_MultinomialDirichlet_"
K_list, K_zz_dict = RCRP_gibbs_sampler!(rcrp, xx, zz,
    n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)

K_hist = hist(K_list, 0.5:maximum(K_list)+0.5)[2]
candidate_K = indmax(K_hist)

pos_components, nn, zz2table = posterior(rcrp, xx, K_zz_dict, candidate_K)


# posterior distributions
inferred_topics = Array(Matrix{Float64}, TT)
for tt = 1:TT
	inferred_topics[tt] = zeros(Float64, length(pos_components[tt]), vocab_size)

	for k_t = 1:length(pos_components[tt])
		inferred_topics[tt][k_t, :] = mean(pos_components[tt][k_t])
	end
end

# visualizing the results
tt = 1
visualize_bartopics(inferred_topics[tt])