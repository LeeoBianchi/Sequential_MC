# Sequential_MC
**A Python implementation of a sequential Monte Carlo method for inferring the parameters of a Markov hidden process.**

Given x1 normally distributed, we have x๐ก = ax๐กโ1 + ๐ + ๐๐ก;  ๐ก = 2,3,...; 
and: ๐ฆ๐ก = 1 if ๐ฅ๐ก < โ0.5, 2 if โ 0.5 โค ๐ฅ๐ก < 0.5, 3 if ๐ฅ๐ก โฅ 0.5

Thus the gaussian field composed by the xs is discretized through the ys, which is the only variable we have data about.

 1) First we want to infer expected values and variances of the xs constituting the gaussian hidden model at each step of the chain.

 2) We want then to perform an online-learning for infering about the parameter a of the model.
