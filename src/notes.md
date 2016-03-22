##Notes

These are the notes that has come to my mind while coding this package.


* the current gibbs sampling method just utilities the last sample of the MCMC chain. Consider implementing other
  strategies as well. like averaging the last N samples and etc. In a finite model it might be as simple as averaging the components of the N last sample (but is it? label switching is not a problem?) In nonparametric methods the number of clusters also changes. How about there? can we think of averaging the last N Gibbs sample that had C clusters and average?


* add help and docstring to Julia functions

* implement a count version for LDA and see which one is faster

* Make sure you have not sued  .==, .< and these kind of operators. They should be avoided for the sake of performance. Use loops Instead. Loops are faster. Loops are better.

* consider using pmap()

* log scale for stick_breaking instead of sticks.

* can I use loglikelihood as a measure of knowing if the chain has been running long enough?

* Although dHDP returns the correct nn, it cannot retrieve rr. Does it have anything to do with identifiability? Is it important at all?

* see if changing the direction of matrices u have affects the speed.

* think of saving the results whenever you want. Maybe pausing for a few secs after finishing each iter
* do I really need to keep track of components in HDP? saving them I mean?
* how to use this learnt model to estimate topic dist for a new doc?

* a function to compute the perplexity

* look at all the code you have written as the for condition can not change in its body. You have to change the for loop to a while loop

* take a look at devetorize.jl by Dahua Lin

* wherever possible use finfnz instead of find. It is faster.

* reorder the BIAS.jl

* write_top_doctopic or LDA

* make sure when you save some results, there is a way to load and continue from the loaded data

* I noticed that CRF_sampler is faster than collapsed_sampler but apparently not as accurate.


* init_zz does it have a description for all models?

TODO
write posterior for RCRP and clean up inference part of demos
