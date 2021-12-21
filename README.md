# Sequential_MC
**A Python implementation of a sequential Monte Carlo method for inferring the parameters of a Markov hidden process.**

Given x1 normally distributed, we have x𝑡 = ax𝑡−1 + 𝜆 + 𝜀𝑡;  𝑡 = 2,3,...; 
and: 𝑦𝑡 = 1 if 𝑥𝑡 < −0.5, 2 if − 0.5 ≤ 𝑥𝑡 < 0.5, 3 if 𝑥𝑡 ≥ 0.5

 1) First we want to infer expected values and variances of the x, normal distributed variable, constituting the gaussian hidden model at each step of the chain.

 2) We want then to perform an online-learning for infering about the parameter a of the model.
