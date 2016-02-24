#=
cgcbib_abs_LDA.jl

Latent Dirichlet Allocation for cgcbib dataset. This is the dataset
Yee Whye Teh etal used in their original HDP paper (Teh etal 2004)

14/10/2015
Adham Beyki, odinay@gmail.com
=#


using BIAS


## --- reading the data --- ##
f     = open("datasets\\cgcbib\\cgcbib_abs_bow.txt")
lines = readlines(f)
close(f)

n_lines   = length(lines)
n_groups  = 5957
n_group_j = zeros(Int, n_groups)
n_vocab   = 3793

lines_mat = zeros(Int, n_lines, 4)
for i = 1:n_lines
    line = split(strip(lines[i]))
    line = [parse(Int, s) for s = line]
    lines_mat[i, :] = line
end

xx = Array(Vector{Int}, n_groups)

for jj = 1:n_groups
    print("$jj ")
    mask = lines_mat[:, 1] .== jj-1
    xx_jj_mat = lines_mat[mask, :]
    xx_jj = Int[]

    for ii=1:size(xx_jj_mat, 1)
        for rr=1:xx_jj_mat[ii, 4]
            push!(xx_jj, xx_jj_mat[ii, 3]+1)
        end
    end
    xx[jj] = xx_jj
end

n_group_j = zeros(Int, n_groups)
for jj = 1:n_groups
    n_group_j[jj] = length(xx[jj])
end



## ------- inference -------- ##
# constructing the conjugate component
dd = n_vocab
aa = 0.5 * n_vocab
q0 = MultinomialDirichlet(dd, aa)


# constructing the LDA model
lda_KK = 70
lda_aa = 1.5
lda    = LDA(q0, lda_KK, lda_aa)

# initializing the cluster labels
zz = Array(Vector{Int}, n_groups)
for jj = 1:n_groups
    zz[jj] = ones(Int, n_group_j[jj])
end
init_zz!(lda, zz)


# sampling
n_burnins   = 200
n_lags      = 1
n_samples   = 300
sample_hyperparam = true
store_every = 1000
filename    = "cgc_LDA_results_"

collapsed_gibbs_sampler!(lda, xx, zz, n_burnins, n_lags, n_samples, store_every, filename)

# posterior distribution
posterior_components, nn = posterior(lda, xx, zz)


f = open("datasets\\cgcbib\\cgcbib_abs_vocab.txt")
vocab = readlines(f)
close(f)
vocab = [strip(vv) for vv in vocab]


dirname = "datasets\\cgcbib\\LDA_results"
mkdir(dirname)

for kk = 1:length(posterior_components)

  filename = join([dirname, "topic$(kk).csv"], "/")
  topic2csv(filename, vocab, posterior_components[kk].alpha)

  filename = join([dirname, "topic$(kk).png"], "/")
  topic2png(filename, vocab, posterior_components[kk].alpha)
end
