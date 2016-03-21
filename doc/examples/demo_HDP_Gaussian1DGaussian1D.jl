#=
demo_LDA_Gaussian1DGaussian1D

A demo for LDA with Gaussian1DGaussian1D Bayesian components.

28/07/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS
srand(123)

## --- synthesizing the data --- ##
true_gg = 1.0
true_aa = 0.5
n_groups = 300
n_group_j = 100 * ones(Int, n_groups)
join_tables = true

true_tji, true_njt, true_kjt, true_nn, true_mm, true_zz, true_KK = BIAS.gen_CRF_data(n_group_j, true_gg, true_aa, join_tables)

vv = 0.001			# fixed variance
ss = 2
true_atoms = [Gaussian1D(ss*kk, vv) for kk = 1:true_KK]

xx = Array(Vector{Float64}, n_groups)
for jj = 1:n_groups
    xx[jj] = zeros(Float64, n_group_j[jj])
    for ii = 1:n_group_j[jj]
        kk = true_zz[jj][ii]
        xx[jj][ii] = sample(true_atoms[kk])
    end
end

## ------- inference -------- ##
# constructing the Bayesian component of LDA model
m0 = mean(mean(xx))
v0 = 10.0
q0 = Gaussian1DGaussian1D(m0, v0, vv)

# constructing the HDP model
hdp_KK_init = 1
hdp_gg = 10.0
hdp_g1 = 0.1
hdp_g2 = 0.1
hdp_aa = 5.0
hdp_a1 = 0.1
hdp_a2 = 0.1
hdp = HDP(q0, hdp_KK_init, hdp_gg, hdp_g1, hdp_g2, hdp_aa, hdp_a1, hdp_a2)

# sampling
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
	zz[jj] = ones(Int, n_group_j[jj])
end
init_zz!(hdp, zz)

n_burnins   = 100
n_lags      = 2
n_samples   = 200
sample_hyperparam = true
n_internals = 10
store_every = 100
filename    = "demo_HDP_Gaussian1DGaussian1D_"


# KK_list, KK_dict, betas, gammas, alphas = collapsed_gibbs_sampler!(hdp, xx, zz, n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)
KK_list, KK_dict = CRF_gibbs_sampler!(hdp, xx, zz, n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)

# posterior distributions
KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_KK = indmax(KK_hist)

# posterior_components, nn, pij = posterior(hdp, xx, KK_dict, candidate_KK)
pos_components, tji, njt, kjt, zz, nn, mm, pij = posterior(hdp, xx, KK_dict, candidate_KK, join_tables)