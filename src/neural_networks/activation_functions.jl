module ActivationFunctions

"""
    sigmoid(x)

Computes the sigmoid activation of scalar `x`, returning a scalar.
"""
sigmoid(x::Number)::Number = (1.0 + exp(-x))^(-1)

"""
    sigmoid(x)

Computes the sigmoid activation of vector `x`, returning a vector.
"""
sigmoid(x::AbstractVector{<:Number})::AbstractVector{<:Number} = sigmoid.(x)

"""
    relu(x)

Computes the rectified linear unit activation of scalar `x`, returning a scalar.
"""
relu(x::Number)::Number = max(0.0, x)

"""
    relu(x)

Computes the rectified linear unit activation of vector `x`, returning a vector.
"""
relu(x::AbstractVector{<:Number})::AbstractVector{<:Number} = max.(0.0, x)

"""
    softmax(x)

Computes the softmax activation of vector `x`, returning a vector.

Unlike some other activation functions, softmax does not treat vector elements
independently.
"""
softmax(x::AbstractVector{<:Number})::AbstractVector{<:Number} =
    exp.(x .- maximum(x)) .* sum(exp.(x .- maximum(x)))^(-1)

end # module
