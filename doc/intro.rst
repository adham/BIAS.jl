.. _section-Introduction:

Introduction
============
A few sentences on
    * mixture modeling
    * Bayesian n treatment of mixture models
    * why hierarchical mixture models
    * why temporal


Mixture modelling is a probabilistic representations of subpopulations (or clusters) within a population. It provides a flexible framework for statistical modeling and analysis.


Mixture Models
--------------
The pdf/pmf a mixture distributino is given by convex combination of the pdf/pmf of its individual components. We say a distribution :math:`f` is a mixture of :math:`K` component distributions if

.. math::

    f\left(x;\Theta,\pi\right)=\sum_{k=1}^{K}\pi_{k}f\left(x;\theta_{k}\right)


A mixture model is characterized by a set of component parameters :math:`\Theta=\left\{ \theta_{1},\ldots,\theta_{K}\right\}` and a prior distribution :math:`\pi` over these components.