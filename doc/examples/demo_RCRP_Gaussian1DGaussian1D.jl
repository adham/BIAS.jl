#=
demo_RCRP_Gaussian1DGaussian1D.jl

demo for Recurrent Chinese Restaurant Process with
Gaussiian1DGaussian1D conjugate component.

19/01/2015
Adham Beyki, odinay@gmail.com
=#

using BIAS

srand(100)

## --- synthesizing the data --- ##

TT = 10
N_t = fill(500, TT)
true_aa = 0.5

true_nn, true_zz = BIAS.gen_RCRP_data(N_t, true_aa)

true_KK = size(true_nn, 2)

# constructing the topics
scale = 10.0
vv = 0.01
true_atoms = [Gaussian1D(scale*kk, vv) for kk = 1:true_KK]


# constructing the observations
xx = Array(Vector{Float64}, TT)
for tt = 1:TT
    xx[tt] = zeros(Float64, N_t[tt])
    for ii = 1:N_t[tt]
        kk = true_zz[tt][ii]
        xx[tt][ii] = sample(true_atoms[kk])
    end
end


## ------- inference -------- ##

# constructing the conjugate
m0 = mean(xx[1])
v0 = 2
q0 = Gaussian1DGaussian1D(m0, v0, vv)

# constructing thee model
init_KK = 1
rcrp_aa = 1
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
n_samples   = 300
sample_hyperparam = true
n_internals = 10
store_every = 50
filename    = "demo_RCRP_Gaussian1DGaussian1D_"
KK_list, KK_zz_dict = RCRP_gibbs_sampler!(rcrp, xx, zz,
    n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)


# posterior distributions
KK_hist = hist(KK_list, 0.5:maximum(KK_list)+0.5)[2]
candidate_KK = indmax(KK_hist)

pos_components, nn, zz2table = posterior(rcrp, xx, KK_zz_dict, candidate_KK)