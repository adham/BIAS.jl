.. index:: Examples; RCRP for Mltinomial likelihood

.. _example-RCRP-MultinomialDirichlet:

RCRP for Multinomial likelihood
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

    N_t = fill(500, TT)
    n_tokens = 25
    vocab_size = 9

    true_topics = BIAS.gen_bars(true_KK, vocab_size, 0.0)



    xx = Array(Vector{Sent}, TT)
    true_zz = Array(Vector{Int}, TT)
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
        xx_ = Array(Sent, N_t[tt])
        true_zz_ = ones(Int64, N_t[tt])

        for n = 1:N_t[tt]
            kk = sample(mix)
            kk = epoch_chains[tt][kk]
            true_zz_[n] = kk
            sentence = sample(true_topics[kk, :][:], n_tokens)
            xx_[n] = BIAS.sparsify_sentence(sentence)
            true_nn[tt, kk] += 1
        end

        xx[tt] = xx_
        true_zz[tt] = true_zz_
    end




    dd = vocab_size
    aa = 1.0
    q0 = MultinomialDirichlet(dd, aa)

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
    n_lags      = 3
    n_samples   = 400
    sample_hyperparam = true
    n_internals = 10
    store_every = 10000
    filename    = "demo_RCRP_MultinomialDirichlet"
    nn, components, zz2table = collapsed_gibbs_sampler!(rcrp, xx, zz,
        n_burnins, n_lags, n_samples, sample_hyperparam, n_internals, store_every, filename)



Result
^^^^^^
.. code-block:: julia

julia> z2table
5x8 Array{Int64,2}:                   
 0  1  2  3  0  0  0  0               
 0  1  2  3  4  0  0  0               
 0  1  2  0  3  4  0  0               
 0  0  1  0  2  3  4  0               
 1  0  4  0  2  6  3  5)              
                                      
                                      
julia> nn                             
5x8 Array{Int64,2}:                   
   0  162  170  168    0    0    0  0 
   0  136  123  120  121    0    0  0 
   0  122  127    0  108  143    0  0 
   0    0  133    0  125  131  111  0 
 126    0  115    0  118    6  133  2 
          
                                      
julia> true_nn                        
5x6 Array{Int64,2}:                   
 168  162  170    0    0    0         
 120  136  123  121    0    0         
   0  122  127  108  143    0         
   0    0  133  125  131  111         
   0    0  115  118  132  135         
                                      
