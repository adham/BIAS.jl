module BIAS

import Distributions
import JLD
import MAT


export
Gaussian1D, Dirichlet,
Gaussian1DGaussian1D, MultinomialDirichlet,
BMM, DPM, LDA, HDP, dHDP, RCRP,
init_zz!, collapsed_gibbs_sampler!, truncated_gibbs_sampler!, CRF_gibbs_sampler!, RCRP_gibbs_sampler!,
posterior, sample, pdf, logpdf, visualize_bartopics, topic2png, topic2csv,
Sent

include("common.jl")
include("viz_utility.jl")
include("distributions.jl")
include("conjugates.jl")
include("BMM.jl")
include("LDA.jl")
include("HDP.jl")
include("RCRP.jl")
include("generate_data.jl")
end
