In this project, I attempt to reproduce the simulation study implemented in 
"METHODS FOR EPIDEMIOLOGIC ANALYSES OF MULTIPLE EXPOSURES: A REVIEW AND
COMPARATIVE STUDY OF MAXIMUM-LIKELIHOOD, PRELIMINARY-TESTING, AND 
EMPIRICAL-BAYES REGRESSION*" by Sander Greenland (1996). The paper compares 
multiple  regression approaches for a logistic model on 
epidemiological-inspired data. Four methods are considered: maximum 
likelihood estimation, parametric empirical Bayes, 
semi-Bayes, and preliminary testing (Greenland, 1996 pg 6).

## Aims:
To compare the performance of estimating regression effects in an 
epidemiological setting with binary outcomes between standard epidemiological
regression methods (maximum likelihood estimation and preliminary testing) 
and two empirical Bayes hierarchical modeling methods.

## Data Generating Mechanisms:
The data generating process described in the paper is a
hierarchical model where the first layer (stage 1) is a
logistic model connecting the mean of binary outcome
data $y$ to a linear predictor through a logit link
function. The linear predictor is a set of covariates
$X$ multiplied by a vector of regression coefficients
$\beta$. $\beta$ itself is then modeled in stage 2
as a linear function $\beta_i = z_i \pi_i + \delta_i$
where the $z_i$ are $p \times 1 $ vectors of prior
covariates, $pi$ is a vector of prior coefficients,
and $\delta_i$ is standard Gaussian noise.

More specifically, the data is generated as follows:

### Data Generation Details
- Generation of $\beta_i = z_i \pi_i + \delta_i$, where
$z_i \sim Ber(0.2), \pi_i \sim Exp(\tau_1)$, and
$\delta_i \sim N(0, \tau_0^2)$.

- Generation of $N \times n$ matrix of covariates $X$:
    1. First, sample rows of $X$ from $N(0, \Sigma)$, where
    $\Sigma$ has diagonal elements 1 and off diagonal elements $r$.

    2. Secondly, draw $n$ observations from $U(-0.25, 0.25)$

    3. Thirdly, dichotomize each column of X's based off of
    the corresponding draw in step 2 (1 for values larger
    than the threshold, 0 otherwise).

- Generation of $y$: Sample $y$ from Bernoulli distributions such that
$P(y_k = 1) = 1/(1+ exp(-\alpha - x_k\beta - \epsilon_k))$
where $\epsilon_k$ are i.i.d $N(0, \sigma^2)$. Set the parameter
$\alpha$ to the mean of $x_k\beta + \epsilon_k$

### Parameters
The parameter values included in the simulations are as follows:
- $\tau_0 = \tau_1 = 0.2$
- $\sigma = 0.5$
- $r = 0.5$
- $n =$ 4, 10, or 20
- $N = $ 40, 100, or 500 if $n = 4$ or $n = 10$,
  and $N = $ 100, 500, 2000 for $n = 20$.
- The number of iterations was set to $8000/n$ when
  $N \geq 100$ and $16000/n$ when $N = 40$.

There were also simulations conducted by setting
default values and then varying several parameters individually.
The default values were $N = 100, n = 10, \tau_0 = \tau_1 = 0.2$,
$\sigma = 0.5, r = 0.5$, and $E(\sum_k Y_k) = N/2$. 

The parameter settings which were varied individually are:
- $r = 0, 0.9$
- $E(\sum_k Y_k) = N/4$
- $\tau_0 = \tau_1 = 0$
- $\tau_0 = 0.4, \tau_1 = 0$
- $\tau_0 = 0, \tau_1 = 0.4$
- $\tau_0 = \tau_1 = 0.4$
- $\sigma = 2$

Additionally, for one simulation, $X$ was generated with Gaussian rows 
instead of Bernoulli by removing the dichotomization step.

## Estimands:
Estimates $n \times 1$ vector or regression coefficients $\beta$,
and a 95% confidence interval for $\beta$, computed as
$\hat{\beta} \pm 1.96*\text{se}(\hat{\beta})$. The calculations
for the standard errors are discussed in the Methods section.

## Methods:

### MLE: 
Fit a logistic GLM with linear predictor
$\eta_k = \alpha + x_k \beta$ using IRLS. Stepsizes
are halved at each iteration.

### Parametric Empirical Bayes
Estimation is conducted in two stages.
First, MLE is performed on the stage 1 model to obtain $\hat{\beta}$
and the inverse of the observed information matrix at $\hat{\beta}$:
$\hat{V}$. Then, stage 2 model estimates are computed using method of moments
estimators. Specifically, the following system of 7 equations are solved:

$$
\begin{aligned}
W^* = & (\hat{V} + \tilde{\tau}^2I)^{-1} //
\mu^* = & Z\pi^* //
\tilde{\tau}^2 = & nR/(n-p) - \bar{V}^* //
R = & e^TW^*e/(\sum_{ij} W_{ij}^*) //
\pi^* = & (Z^T W^* Z)^{-1}Z^T W^* \hat{\beta} //
\bar{V}^* = & W^* \hat{V}/(\sum_{ij} W_{ij}^*) //
e = & \hat{\beta} - \mu^* //
\end{aligned}
$$
and then $B^* = (n-p-2)W^*\hat{V}/(n-p)$ is computed.
Then, the mean of the posterior distribution is the final estimate of
$\beta$, approximated by $\beta^* = B^* \mu^* + (I-B^*)\hat{\beta}$.
The variance of $\beta^*_i$ was approximated by
$$
v_i^* = \hat{V}_{ii} - (1 - H^*_{ii})(\hat{V}B^*)_{ii} + 
(\bar{V}^*_{ii} + \tilde{\tau}^2 I)W^*_{ii}A_{ii}
$$
where $H^* = Z(Z^T W^* Z)^{-1} Z^T W^*$ and
$A = 2B^*e(B^* e)^T/(n-p)$.

### Semi-Bayes
Estimators are computed similar to the empirical Bayes approach,
however, $\tau^2$ is fixed to a specific value (either 0.5, 1, or 2 times the
true value). Hence, a different system of equations is solved:

$$
\begin{aligned}
\tilde{\mu} = & Z\tilde{\pi} \\
\tilde{\pi} = & (Z^T W Z)^{-1}Z^T W \hat{\beta}
\end{aligned}
$$
where $\hat{\beta}$ and $\hat{V}$ are obtained via MLE as before.
We then compute $B = W\hat{V}$ and the approximate posterior mean
$\tilde{\beta} = B\tilde{\mu} + (I - B)\hat{\beta}$. The approximate
variance of $\tilde{\beta}_i$ is computed by 
$$
v_i = \hat{V}_{ii} - (1-\tilde{H}_{ii})(\hat{V}B)_{ii}
$$
where $\tilde{H} = (Z^T W Z)^{-1}Z^T W$.

### Preliminary Testing
First, regression coefficient estimates
$\hat{\beta}$ were computed using MLE. Then, Wald-type statistics
$\hat{\beta}_i / SE(\hat{\beta}_i)$. Predictors were removed
from the model if the corresponding regression coefficient had
a Wald statistic smaller than 0.1. After removing predictors,
the model was refit using MLE. Confidence intervals for the
resulting coefficients are computed using the information
matrix from the full model, but evaluated at the coefficient
estimates from the reduced model.

Note that trials were discarded if the ML estimates were infinite.

## Performance Measures
To assess the performance of the estimation
methods, the coverage rate and mean lengths of the 95% confidence 
intervals are computed. Additionally, the RMSE of the point
estimates is also computed

Other "miscellaneous measures" are calculated, such as the
percentage of trials where the empirical Bayes procedure
estimated $\tau$ to be zero, the percentage of trials where
the preliminary testing algorithm selected zero predictors,
and the average percent of variables selected by the preliminary
testing algorithm. These were computed for each simulation
scenario. 