# ReForecasts.jl
**Package is under construction and currently offers limited functionality**

**ReForecasts.jl** implements methods for handling and reconciling hierarchical forecasts

## Quick Start

### Installation
Install the package with:
```julia
using Pkg
Pkg.add(url="https://github.com/lipiecki/ReForecasts.jl")
```
and load it into the namespace:
```julia
using ReForecasts
```

### Example
Let `observations::Matrix` and `forecasts::Matrix` be matrices of observed and predicted time series values, with rows corresponding to time steps and columns corresponding to different aggregation levels. Let `b::Int` be a number of bottom level series. 

Infer the hierarchy from observations, i.e., reconstruct the summing matrix:
```julia
hierarchy = Hierarchy(observations, b)
```

Create a `LinearReconciler` by specifying the method and hierarchy, here we use `:mint` for minimum trace reconciliation [(Wickramasuriya et al. 2019)](https://doi.org/10.1080/01621459.2018.1448825):
```
reconciler = LinearReconciler(:mint, hierarchy)
```

Divide the data into training and testing:
```
train_forecasts = forecasts[1:100, :]
train_observations = observations[1:100, :]
test_forecasts = forecasts[101:end, :]
test_observations = observations[101:end, :]
```

Fit the reconciler to the training data:
```
fit(reconciler, train_forecasts, training_observations, hierarchy)
```

Reconcile the forecasts in the testing period
```
reforecasts = reconciler(test_forecasts)
```

### Avaiable methods
`LinearReonciler(method::Symbol, hierarchy::Hierarchy)` currently allows the following options for the `method` argument:
- `:bu` - Bottom-Up reconciliation
- `:ols` - Ordinary Least Squares reconciliation
- `:mint` - Minimum Trace reconciliation [(Wickramasuriya et al. 2019)](https://doi.org/10.1080/01621459.2018.1448825)
- `:shrinkmint` - Minimum Trace reconciliation with covariance shrinkage by [Schäfer & Strimmer (2005)](https://doi.org/10.2202/1544-6115.1175)
- `:honeymint` - Minimum Trace reconciliation with covariance shrinkage by [Ledoit & Wolf (2004)](https://doi.org/10.3905/jpm.2004.110)
- `:icomb` - Information Combination [(Nguyen et al. 2026)](https://arxiv.org/abs/2605.29611)
- `:ridgeicomb` - Information Combination with ridge regularization [(Nguyen et al. 2026)](https://arxiv.org/abs/2605.29611)
