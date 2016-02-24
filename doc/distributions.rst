.. _section-Distributions:

Distributions
=============
All distributions provided in this package are organized in a type hierarchy as follows:

.. code-block:: julia

    abstract Distribution
    distribution <: Distribution


Here is the list of implemented distribution. This list will grow as I continue to develop this package.

- Gaussian1D
- Dirichlet


Gaussian1D
----------
Gaussian1D is the univariate Gaussian distribution. It is parametrized by its mean and variance.

.. code-block:: julia

    julia> using BIAS
    julia> srand(123)

    julia> qq = Gaussian1D(5, 2)
    Gaussian1D distribution
    mean: mu=5.0, variance: vv=2.0

    julia> sample(qq)
    6.683292

    julia> sample(qq, 100)
    100-element Array{Float64, 1}:
    7.89656
    6.61595
    ...

    julia> pdf(qq, 4.5)
    0.265003

    julia> logpdf(qq, 4.5)
    -1.328012


Dirichlet
---------
Dirichlet distribution is a probability distribution on a probability simplex where its draws are multinomial random variables.

.. code-block:: julia


    julia> qq = Dirichlet([0.2, 0.2, 0.2, 0.2])
    Dirichlet distribution
    cardinality = 4
    alpha       = [0.2,0.2,0.2,0.2]

    julia> qq = Dirichlet(5, 4)
    Dirichlet distribution
    cardinality = 4
    alpha       = [1.25,1.25,1.25,1.25]

    julia> mean(qq)
    4-elemnt Array{Float64, 1}
    0.25
    0.25
    0.25
    0.25

.. caution:: pdf and logpdf need to be revised.



