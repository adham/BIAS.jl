#=
conjugates.jl

25/08/2015
Adham Beyki, odinay@gmail.com

TODO:
    @check_args
=#

abstract Conjugate

#=
##################################
###### Gaussian1DGaussian1D ######
##################################
Gaussian1DGaussian1D constructs a Bayesian likelihood-prior pair with
a Gaussian1D likelihood with known variance, where its mean is endowed
with another Gaussian1D.

Parameters:
    m0      (1x1) mean hyper-parameter of the Gaussian1D prior
    v0      (1X1) variance hyper-parameter of the Gaussian1D prior
    vv      (1x1) fixed (known) variance of the likelihood

MODEL:
    mu | m0, v0         ~ Gaussian1D(m0, v0)
    x_1,...,x_n | mu,vv ~ Gaussian1D(mu, vv)
=#
type Gaussian1DGaussian1D <: Conjugate
    m0::Float64             # mean hyper-parameter of the prior
    v0::Float64             # variance hyper-parameter of the prior

    mu::Float64             # mean of the likelihood
    vv::Float64             # fixed variance of the likelihood

    nn::Int                 # number of data points the component contains

    Gaussian1DGaussian1D(m0::Float64, v0::Float64, mu::Float64, vv::Float64, nn::Int) = new(m0, v0, mu, vv, nn)
end

Gaussian1DGaussian1D(m0::Real, v0::Real, vv::Real) = Gaussian1DGaussian1D(
        convert(Float64, m0),
        convert(Float64, v0),
        convert(Float64, m0),
        convert(Float64, vv),
        0)
Base.deepcopy(me::Gaussian1DGaussian1D) = Gaussian1DGaussian1D(me.m0, me.v0, me.mu, me.vv, me.nn)


function Base.show(io::IO, me::Gaussian1DGaussian1D)
    println(io, "Gaussian1DGaussian1D conjugate component")
    println(io, "likelihood parameters: mu=", me.mu, ", vv=", me.vv)
    println(io, "prior parameters     : m0=", me.m0, ", v0=", me.v0)
    println(io, "number of data points: nn=", me.nn)
end


# adding/removing data points to/from the conjugate component
function additem!(me::Gaussian1DGaussian1D, xx::Float64)
    me.nn += 1
    me.mu = ((me.nn-1)/me.nn)*me.mu + xx/me.nn
end
function delitem!(me::Gaussian1DGaussian1D, xx::Float64)
    if me.nn < 1
        throw("there is no more datapoints in the conjugate component")
    elseif me.nn == 1
        me.mu = me.m0
        me.nn = 0
    else
        me.mu = me.mu*(me.nn/(me.nn-1)) - xx/(me.nn-1)
        me.nn -= 1
    end
end


function posterior(me::Gaussian1DGaussian1D)

    if me.nn>0
        denom = me.vv + me.nn*me.v0
        mu = (me.nn * me.v0 * me.mu + me.vv * me.m0) / denom
        vv = me.v0 * me.vv / denom
    else
        mu = me.m0
        vv = me.v0
    end

    Gaussian1D(mu, vv)
end


function logpredictive(me::Gaussian1DGaussian1D, xx::Float64)
    # log p(xx|x1,...,xn)

    post = posterior(me)
    mu = post.mu
    vv = post.vv

    dummy = vv + me.vv
    val = -(xx - mu)^2 / (2 * dummy) - 0.5 * log(2 * pi * dummy)

    val
end

loglikelihood(me::Gaussian1DGaussian1D, xx::Float64) = logpdf(posterior(me), xx)



##################################
###### MultinomialDirichlet ######
##################################
#=
MultinomialDirichlet constructs a Bayesian likelihood-prior pair with a Multinomial likelihood
where its probability vector is drawn from a symmetric Dirichlet prior.

Parameters:
    dd: (1x1)  Cardinality, i.e. length of the multinomial probability vector
    pp: (ddx1) Multinomial probability vector
    aa: (1x1)  Concentration parameter of the Dirichlet prior

    mm: (1x1)  total count in xx
    mi: (ddx1) individual counts
    nn: (1x1)  number of data points

Model:

    pp | aa                 ~ Dirichlet(aa/dd,...,aa/dd)
    x_1,...,x_n | mm, mi, pp ~ Multinomial(mm, mi, pp)
=#

type MultinomialDirichlet <: Conjugate
    dd::Int                               # Cardinality
    aa::Float64                           # Dirichlet concentration hyper-parameter
    pp::Vector{Float64}                   # Multinomial probability vector

    mm::Int                               # total count in data
    mi::Vector{Int}                       # individual count in data
    nn::Int                               # number of data points

    MultinomialDirichlet(dd::Int, aa::Float64, pp::Vector{Float64}, mm::Int, mi::Vector{Int}, nn::Int) = new(dd, aa, pp, mm, mi, nn)
end
MultinomialDirichlet(dd::Int, aa::Float64) = MultinomialDirichlet(dd, aa/dd, zeros(Float64, dd), 0, zeros(Int, dd), 0)
Base.deepcopy(me::MultinomialDirichlet) = MultinomialDirichlet(me.dd, me.aa, deepcopy(me.pp), me.mm, deepcopy(me.mi), me.nn)

# displaying the MultinomialDirichlet component
function Base.show(io::IO, me::MultinomialDirichlet)
    println(io, "MultinomialDirichlet component")
    println(io, "cardinality: dd=$(me.dd), Dirichlet prior parameter: aa=$(me.aa)")
    println(io, "data: mm=$(me.mm), nn=$(me.nn)")
end

# adding/removing data points to/from the component
function additem!(me::MultinomialDirichlet, xx::Int)
    me.mi[xx] += 1
    me.mm += 1
    me.nn += 1
end
function delitem!(me::MultinomialDirichlet, xx::Int)
    me.mi[xx] -= 1
    me.mm -= 1
    me.nn -= 1
end
function additem!(me::MultinomialDirichlet, xx::Sent)
    xx_len = length(xx.w)
    @inbounds for ii = 1:xx_len
        me.mi[xx.w[ii]] += xx.c[ii]
        me.mm += xx.c[ii]
    end
    me.nn += 1
end
function delitem!(me::MultinomialDirichlet, xx::Sent)
    xx_len = length(xx.w)
    @inbounds for ii = 1:xx_len
        me.mi[xx.w[ii]] -= xx.c[ii]
        me.mm -= xx.c[ii]
    end
    me.nn -= 1
end

logpredictive(me::MultinomialDirichlet, xx::Int) = log((me.aa + me.mi[xx]) / (me.aa*me.dd + me.mm))
function logpredictive(me::MultinomialDirichlet, xx::Sent)
    xx_len = length(xx.w)
    lll = me.aa + me.mi
    mm = 0
    mi = zeros(Int, xx_len)
    @inbounds for ii = 1:xx_len
        lll[xx.w[ii]] += xx.c[ii]

        mm += xx.c[ii]
        mi[ii] = xx.c[ii]
    end

    ll = lgamma(mm+1) - sum(lgamma(mi+1)) + lgamma(me.aa*me.dd + me.mm) -
            lgamma(me.aa*me.dd + me.mm + mm) +
            sum(lgamma(lll) - lgamma(me.aa+me.mi))

    ll
end

function posterior(me::MultinomialDirichlet)
    aa = me.mi + me.aa
    return Dirichlet(aa)
end

loglikelihood(me::MultinomialDirichlet, xx::Int) = mean(posterior(me))[xx]
function loglikelihood(me::MultinomialDirichlet, xx::Sent)
    ll = 0.0
    alpha = mean(posterior(me))
    @inbounds for i=1:length(xx)
        ll += xx.c[i] * alpha[xx.w[i]]
    end
    ll
end

