BIAS: Bayesian Inference At Scale in Julia
===========================================

:Version: |release|
:Requires: julia releases 0.4.0 or later
:Date: |today|
:Author: Adham Beyki (abeyki@deakin.edu.au)
:Web site: http://github.com/adham/BIAS.jl
:License: `MIT <https://github.com/adham/BIAS.jl/blob/master/LICENSE.md>`_


Purpose
-------
**BIAS** is an open source implementation of several Bayesian parametric and non-parametric hierarchical mixture models in `julia <http://julialang.org>`_. The implemented models include:

1. Bayesian Mixture Model (BMM)
2. Latent Dirichlet Allocation (LDA)
3. Dirichlet Process Mixture Model (DPM)
4. Hierarchical Dielectric Mixture Model (HDP)
5. dynamic Hierarchical Dirichelt Mixture Model (dHDP)
6. Recurrent Chinese Restaurant Process (RCRP) also known as Temporal Dirichelt Process Mixture model (TDPM)


Getting Started
---------------
The following **julia** command will install the package:

.. code-block:: julia

    julia> Pkg.clone("git://github.com/adham/BIAS.jl.git")


Contents
--------
.. toctree::
    :maxdepth: 2

    intro.rst
    distributions.rst
    conjugates.rst
    models.rst
    examples.rst
    xrefs.rst


Indices
^^^^^^^
* :ref:`genindex`
