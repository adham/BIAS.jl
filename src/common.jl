#=
common.jl

Adham Beyki, odinay@gmail.com
=#


abstract MixtureModel
abstract hMixtureModel


immutable Sent
    w::Vector{Int64}
    c::Vector{Int64}
    Sent(w::Vector{Int64}, c::Vector{Int64}) = new(w, c)
end
Sent(n::Int64) = Sent(zeros(Int64, n), zeros(Int64, n))
Base.length(x::Sent) = length(x.w)



"""
initializes the model cluster assignments
"""
function init_zz!(me::MixtureModel, zz::Vector{Int64})
  zz[:] = rand(1:me.KK, length(zz))
end
function init_zz!(me::hMixtureModel, zz::Vector{Vector{Int64}})
  n_groups = length(zz)
    n_group_j = [length(zz[jj]) for jj = 1:n_groups]
    for jj = 1:n_groups
        zz[jj] = rand(1:me.KK, n_group_j[jj])
    end
end




"""
normalizes a vector with logarithmic scale values
"""
function lognormalize!(pp::Vector{Float64})
    pp_len = length(pp)
    pp_max = maximum(pp)
    @inbounds for kk = 1:pp_len
        pp[kk] = exp(pp[kk] - pp_max)
    end
    pp_sum = sum(pp)
    @inbounds for kk = 1:pp_len
        pp[kk] /= pp_sum
    end
end
function lognormalize!(pp::Vector{Float64}, KK::Int64)
    pp_max = maximum(pp)
    @inbounds for kk = 1:KK
        pp[kk] = exp(pp[kk] - pp_max)
    end
    pp_sum = sum(pp[1:KK])
    @inbounds for kk = 1:KK
        pp[kk] /= pp_sum
    end
end



"""
draws samples from a probability distribution
"""
function sample(w::Vector{Float64})
    r = rand()
    n = length(w)
    i = 1
    cw = w[1]
    while cw < r && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end
function sample(w::Vector{Float64}, n::Int64)
    ret = zeros(Int64, n)
    @inbounds for i = 1:n
        ret[i] = sample(w)
    end
    return ret
end




"""
converts a vector of single observations (ie words) into one sparse representation (Sent type)
"""
function sparsify_sentence(x)
    x_set = unique(x)
    n = length(x_set)
    sent = Sent(n)
    @inbounds for i = 1:n
        sent.w[i] = x_set[i]
        sent.c[i] = countnz(x .== x_set[i])
    end
    return sent
end

rand_Dirichlet(alpha::Vector{Float64}) = Distributions.rand(Distributions.Dirichlet(alpha))
rand_Dirichlet(alpha::Vector{Float64}, n) = Distributions.rand(Distributions.Dirichlet(alpha), n)



"""
constructs weights based on truncated Stick-Breaking process
"""
function stick_breaking(vv::Vector{Float64})
  KK = length(vv)
  pp = ones(Float64, KK)

  pp[2:KK] = 1-vv[1:KK-1]
  pp = vv .* cumprod(pp)
  pp
end
"""
constructs weights based on truncated Stick-Breaking process
"""
function log_stick_breaking(vv::Vector{Float64})
  KK = length(vv)
  pp = zeros(Float64, KK)

  pp[2:KK] = log(1 - vv[1:KK-1])
  pp = log(vv) + cumsum(pp)
  pp
end


"""
deletes column kk from matrix nn and returns nn
"""
function del_column{T}(nn::Matrix{T}, kk::Int64)
  KK = size(nn, 2)
  mask = 1:KK .!= kk
  return nn[:, mask]
end

"""
adds a column to the rightmost of nn and returns nn
"""
function add_column{T}(nn::Matrix{T})
  r, c = size(nn)
  mm = zeros(T, r, c+1)
  mm[:, 1:c] = nn
  return mm
end


"""
generates bar topics
"""
function gen_bars(n_bars, n_vocab, noise_level)
    KK = round(Int, sqrt(n_vocab))
    bars = zeros(Float64, n_bars, n_vocab)

    for kk in 1:round(Int, n_bars/2)
        b = zeros(KK, KK) + noise_level
        b[kk, :] = ones(KK)
        b /= sum(b)
        bars[kk, :] = b[:]
    end

    for kk in round(Int, n_bars/2)+1 : n_bars
        b = zeros(KK, KK) + noise_level
        b[:, kk-round(Int, n_bars/2)] = ones(KK)
        b /= sum(b)
        bars[kk, :] = b[:]
    end
    bars
end



"""
vertical cumulative sum in reverse order from bottom to top
"""
function reverse_cumsum{T}(nn::Matrix{T})

  J = size(nn, 1)
  mm = zeros(T, J, J)
  mm[J, :] = nn[J, :]
  for jj = 2:J
    mm[J-jj+1, :] = nn[J-jj+1, :] + mm[J-jj+2, :]
  end
  mm
end


"""
writes a topic into a CSV file
"""
function topic2csv(filename, vocab, alpha)

  idx_sorted = sortperm(alpha, rev=true)

  words = Array(Tuple{ASCIIString, Float64}, length(vocab))
  for i = 1:length(idx_sorted)
    words[i] = (vocab[idx_sorted[i]], alpha[idx_sorted[i]])
  end

  csvfile = open(filename, "w")

  for i = 1:length(vocab)
    write(csvfile, join(words[i], ","), "\n")
  end

  close(csvfile)
end


"""
writes top n topics used in a document
"""
function write_top_doctopics(pij, filename, topn=5)
  csvfile = open(filename, "w")

  for kk = 1:size(pij, 2)
    idx = sortperm(pij[:, kk], rev=true)[1:topn]
    insert!(idx, 1, kk)
    write(csvfile, join(idx, ","), "\n")
  end

  close(csvfile)
end


# macro for argument checking
macro check_args(D, cond)
    quote
        if !($cond)
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end