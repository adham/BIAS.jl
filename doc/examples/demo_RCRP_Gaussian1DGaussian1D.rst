.. index:: Examples; RCRP for univariate Gaussian likelihood

.. _example-RCRP-Gaussian1DGaussian1D:

RCRP for univariate Gaussian likelihood
---------------------------------------



Scenario
^^^^^^^^

+---------+-------+-------+-------+-------+-------+-------+
|         | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 |
+=========+=======+=======+=======+=======+=======+=======+
| Epoch 1 |   O   |   O   |   O   |       |       |       |
+---------+-------+-------+-------+-------+-------+-------+
| Epoch 2 |   O   |   O   |   O   |   O   |       |       |
+---------+-------+-------+-------+-------+-------+-------+
| Epoch 3 |       |   O   |   O   |   O   |   O   |       |
+---------+-------+-------+-------+-------+-------+-------+
| Epoch 4 |       |       |   O   |   O   |   O   |   O   |
+---------+-------+-------+-------+-------+-------+-------+
| Epoch 5 |       |       |   O   |   O   |   O   |   O   |
+---------+-------+-------+-------+-------+-------+-------+



.. code-block:: julia

    true_KK = 6
    TT = 5
    N_t = fill(300, TT)

    vv = 0.01
    true_atoms = [Gaussian1D(kk, vv) for kk = 1:true_KK]

    xx = Array(Vector{Float64}, TT)
    true_zz = Array(Vector{Int64}, TT)
    true_nn = zeros(Int, TT, true_KK)

    epoch_chains = Array(Vector{Int}, TT)
    epoch_chains[1] = [1, 2, 3]
    epoch_chains[2] = [1, 2, 3, 4]
    epoch_chains[3] = [2, 3, 4, 5]
    epoch_chains[4] = [3, 4, 5, 6]
    epoch_chains[5] = [3, 4, 5, 6]

    for tt = 1:TT
        n_chains = length(epoch_chains[tt])
        mix = ones(Float64, n_chains)/n_chains
        xx_ = ones(Float64, N_t[tt])
        true_zz_ = ones(Int64, N_t[tt])

        for n = 1:N_t[tt]
            kk = sample(mix)
            kk = epoch_chains[tt][kk]
            true_zz_[n] = kk
            xx_[n] = sample(true_atoms[kk])
            true_nn[tt, kk] += 1
        end

        xx[tt] = xx_
        true_zz[tt] = true_zz_
    end

    m0= mean(xx[1])
    v0 = 2
    q0 = Gaussian1DGaussian1D(m0, v0, vv)

    init_KK = 3
    rcrp_aa = 1
    rcrp_a1 = 1
    rcrp_a2 = 1
    rcrp = RCRP(q0, init_KK, rcrp_aa, TT, rcrp_a1, rcrp_a2)

    # sampling
    zz = Array(Vector{Int64}, TT)
    for tt = 1:TT
      zz[tt] = rand(1:rcrp.K, N_t[tt])
    end

    n_burnins   = 100
    n_lags      = 1
    n_samples   = 400
    sample_hyperparam = true
    n_internals = 10
    store_every = 10000
    filename    = "demo_RCRP_Gaussian1DGaussian1D_"
    nn, components, zz2table = collapsed_gibbs_sampler!(rcrp, xx, zz,
        n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)





Result
^^^^^^

.. code-block:: julia

    julia> nn
    5x12 Array{Int64,2}:
     106  93  100   0   0   0  0  1  0  0  0  0
      76  70   75  78   0   0  0  0  1  0  0  0
      73   1   84  74  64   0  2  0  0  1  1  0
       1   0   67  80  84  68  0  0  0  0  0  0
       0   0   77  69  81  72  0  0  0  0  0  1

    julia> true_nn
    5x6 Array{Int64,2}:
     93  106  101   0   0   0
     70   76   75  79   0   0
      0   77   84  75  64   0
      0    0   68  80  84  68
      0    0   77  70  81  72

    julia> zz2table
    5x12 Array{Int64,2}:
     1  2  3  0  0  0  0  4  0  0  0   0
     1  2  3  5  0  0  0  0  6  0  0   0
     1  6  3  2  4  0  5  0  0  7  8   0
     6  0  4  1  3  2  0  0  0  0  0   0
     0  0  8  4  9  3  0  0  0  0  0  10


     