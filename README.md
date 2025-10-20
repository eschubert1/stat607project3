This project attempts to reproduce the simulation study in "METHODS FOR EPIDEMIOLOGIC ANALYSES OF
MULTIPLE EXPOSURES: A REVIEW AND
COMPARATIVE STUDY OF MAXIMUM-LIKELIHOOD,
PRELIMINARY-TESTING, AND EMPIRICAL-BAYES
REGRESSION*" by Sander Greenland (1996).

The above paper conducted a review of four methods for estimating binary 
outcomes in settings inspired by epidemiological work. The methods 
reviewed were maximum likelihood estimation, empirical Bayes estimation,
a semi-bayes procedure which represents a method in between empirical 
Bayes estimation and a fully Bayesian approach, and a preliminary testing 
approach, which selects predictors via maximum likelihood and then refits 
the maximum likelihood estimates using only the selected subset. This 
project reimplements those four methods, and evaluates their performance 
via the mean coverage rate, mean length, and RMSE of the predictors
in several simulation scenarios.

Quantitatively, the results from this project differ substantially to those
of the paper, where most methods did not achieve the 95% nominal coverage rate.
This was especially the case for the Empirical Bayes approach, which exhibited
terrible performance in some scenarios. However, this may be due to the
difficulty of reproducing the method of moments estimator, since the formulas
given in the paper contained multiple typographical errors and could not be 
directly implemented. With the exception of the Empirical Bayes analysis,
however, qualitatively the results follow a similar pattern, with the
Semi-Bayes approaches tending to slightly outperform the MLE in most
cases.

To run the analysis, simply run ```make all``` into a terminal to run all
simulations, compute metrics, and generate figures and tables. Alternatively,
these functions can be separated into ```make simulate```, ```make analyze```,
and ```make figures```. Running ```make clean``` will remove all generated
datasets, estimates, and figures. Additionally, tests can be run using
```make test```. Required packages can be installed by running 
```make setup```. The entire simulation should take less than 5 minutes.
