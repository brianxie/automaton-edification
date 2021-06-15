module LossFunctions

export mse, cross_entropy

using Statistics

"""
    mse(x, y)

Computes the mean-squared error between `x` and `y`, returning a scalar.

For the multi-sample case, takes the error along columns (dimensions), and
averages along rows (count).

- `x`: Either (1) a vector containing a single prediction, or (2) a matrix where
  rows index predictions in a batch and columns indicate dimensions of the
  prediction.
- `y`: Either (1) a vector containing a single label, or (2) a matrix where rows
  index labels in a batch and columns indicate dimensions of the label.

`x` and `y` should have the same dimensionality.
"""
mse(x, y) = mean(sum((x .- y) .^2, dims=2), dims=1)[1]


"""
    cross_entropy(x, y)

Computes the cross-entropy error between `x` and `y`, returning a scalar.

For the multi-sample case, takes the error along columns (dimensions), and
averages along rows (count).

- `x`: Either (1) a vector containing a single prediction, or (2) a matrix where
  rows index predictions in a batch and columns indicate dimensions of the
  prediction.
- `y`: Either (1) a vector containing a single label, or (2) a matrix where rows
  index labels in a batch and columns indicate dimensions of the label.

`x` and `y` should have the same dimensionality.
"""
cross_entropy(x, y) = mean((-1) * sum(y .* log.(x), dims=2), dims=1)[1]

end # module
