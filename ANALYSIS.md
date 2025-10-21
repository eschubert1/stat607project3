For this project, I spent a substantial amount of time reading the paper and documenting the methods used for generating the data and implementing the methods described in the paper, especially the method of moments estimators for the Empirical Bayes and
Semi-Bayes methods. Nevertheless, I was not able to reproduce the results in Greenland's paper, notably with the Empirical Bayes estimates, which performed terribly in several scenarios, particularly when the sample size was set to 500. For most simulation iterations, the MLE was not able to reach the nominal coverage rate, and consequently the other
methods, which rely on the MLE, also exhibited worse performance. However, my results agree with Greenland that the Semi-Bayes
methods tended to do slightly better than the MLE in most cases.

One possibility for the discrepancy in the results
is that I discovered there were some substantial mistakes in the equations listed in the paper as I was implementing them. I
managed to find similar equations in some of the references, however, I was never completely confident that I exactly reproduced
the method used in Greendland's paper due to these issues. Another issue I noticed is that the MLE solver I was using exhibited
much more variability in its estimates than the results would suggest in Greenland's paper. This variability appeared to be
stemming from numerical issues in estimating and inverting the Hessian. It is possible that the solver in statsmodels is less robust then the one used by Greeland. However, there is also a note in Greenland's paper that MLE simulation
iterations which exhibited infinite variance were discarded. Following this, I also discarded iterations where the Hessian
matrix was not invertible and when variance estimates for the parameters were larger than 100. However, it is not entirely
clear what variance estimate would be considered infinite from the procedure used by Greenland.

In his paper, Greenland argues that his choices for the simulation design were motivated by commonly occurring situations from
epidemiology. I do not have experience working with epidemiological data, so I cannot comment to the extent which this seems
plausible. However, in Greenland's setup, no method correctly specifies the model speaks considerably to the neutrality of the study. With that said, Greenland did present the Semi-Bayes approach in an earlier paper, and the other methods are given
some advantages compared to the MLE. One way this happens is that all the other methods use the maximum likelihood estimates
as a starting point for further analysis, and hence the other resources have more resources available for estimation.
Also, for the Semi-Bayes approaches in particular, the assumed value of the prior variance is set to the true value and
then multiplied by either a factor of 0.5, 1, or 2. These adjustments seem relatively small, and presumably the
Semi-Bayes approaches would perform much worse if the prior variance was more substantially misspecified. This is particularly
important because the prior variance would never be known in practice, and I find it difficult to believe that a
factor of 2 is the extent to which this could be misspecified.

To change this simulation design, I would expand the range of prior variance specifications for the Semi-Bayes
approaches to offer a better understanding of when the method breaks down. I might also expand the range of
covariates to 50 and 100 since it is computationally feasible now and perhaps more relevant for current
practice. Otherwise, I thought this simulation study was well designed and do not feel the need
to change much else.

I reproduced some of the tables in Greenland's paper and also created a zipper plot of the confidence intervals
for the first parameter, $\beta_1$. The zipper plots highlight the superior performance of the Semi-Bayes
estimators across simulations, with coverage rates similar to the MLE and preliminary-testing methods but
with much shorter intervals. It is also evident that the Empirical Bayes estimates did not perform well
at all, and suggests there could be something wrong with my implementation or a component which was
missing from Greenland's design.