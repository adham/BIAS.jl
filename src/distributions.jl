#=
distributions.jl

25/08/2015
Adham Beyki, odinay@gmail.com

# TODO
# revise pdf and logpdf of Dirichlet
=#

abstract Distribution

####################################
############ Gaussian1D ############
####################################
immutable Gaussian1D <: Distribution
    mu::Float64
    vv::Float64

    Gaussian1D(mu::Float64, vv::Float64) = (@check_args(Gaussian1D, vv > zero(vv)); new(mu, vv))
end
Gaussian1D(mu::Real, vv::Real) = Gaussian1D(convert(Float64, mu), convert(Float64, vv))

Base.deepcopy(me::Gaussian1D)  = Gaussian1D(me.mu, me.vv)

# displaying the Gaussian1D component
function Base.show(io::IO, me::Gaussian1D)
    println(io, "Gaussian1D distribution")
    println(io, "mean: mu=", me.mu, ", variance: vv=", me.vv)
end

# sampling
sample(me::Gaussian1D)    = me.mu + sqrt(me.vv)*randn()
sample(me::Gaussian1D, n) = me.mu + sqrt(me.vv)*randn(n)

# pdf
pdf(me::Gaussian1D, x::Float64)    = exp(-(x-me.mu)^2/(2*me.vv)) / (sqrt(2*pi*me.vv))
logpdf(me::Gaussian1D, x::Float64) = log(pdf(me, x))



##############################
######### Dirichlet ##########
##############################
type Dirichlet <: Distribution
    alpha::Vector{Float64}

    Dirichlet(alpha::Vector{Float64}) = new(alpha)
end

Dirichlet(alpha::Vector{Int}) = Dirichlet(convert(Vector{Float64}, alpha))
Dirichlet(alpha::Real, dim::Int) = Dirichlet(fill(alpha/dim, dim))
Base.deepcopy(me::Dirichlet)    = Dirichlet(deepcopy(me.alpha))

# displaying the Gaussian1D component
function Base.show(io::IO, me::Dirichlet)
    println(io, "Dirichlet distribution")
    println(io, "cardinality = ", length(me.alpha))
    println(io, "alpha       = ", me.alpha)
end

Base.mean(me::Dirichlet) = me.alpha / sum(me.alpha)

# pdf and logpdf are in fact expectations
pdf(me::Dirichlet, x::Int)    = mean(me)[x]
logpdf(me::Dirichlet, x::Int) = log(pdf(me, x))