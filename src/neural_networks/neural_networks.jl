module NeuralNetworks

# TODO: Add exports

using Zygote

##
## Data types
##

"""
A single layer of a neural network.
"""
struct Layer
    # Each row corresponds to a neuron in the previous layer, including bias
    # term.
    # Each column corresponds to a neuron in this layer, not including the bias
    # term.
    # Biases for a given layer are implicitly added by the next layer, and so do
    # not show up in dimensionality of the same layer. Thus, the dimensionality
    # of this matrix is (num_prev_outputs + 1) * (num_curr_outputs).
    weights::AbstractMatrix{<:Number}
    # Each scalar-valued (f: R -> R) activation function corresponds to a single
    # neuron in this layer.
    activation_fns::AbstractVector{<:Function}
end

"""
A complete neural network.
"""
struct NeuralNetwork
    # Layers of the network, applied in order.
    layers::AbstractVector{<:Layer}
    # Single loss function f: (R^n * R^n) -> R applied to the network's final
    # layer output.
    loss_fn::Function
end

"""
A data structure containing the values computed by a single layer during the
forward-pass.
"""
struct LayerForwardPass
    # Inputs from the previous layer. Includes bias.
    inputs::AbstractVector{<:Number}
    # Sums of each neuron before activation; same dimensionality as layer
    # output.
    sums::AbstractVector{<:Number}
    # Activation of each neuron; same dimensionality as layer output.
    activations::AbstractVector{<:Number}
end

"""
A data structure containing the partial derivatives of the loss with respect to
a given layer during the backward-pass.
"""
struct LayerBackwardPass
    # Partial derivatives w.r.t. pre-activation sums.
    dL_dS::AbstractVector{<:Number}
    # Partial derivatives w.r.t. each weight.
    dL_dW::AbstractMatrix{<:Number}
    # Partial derivatives w.r.t. inputs. This gets used by the previous layer in
    # backprop (these inputs are exactly the outputs of the previous layer).
    dL_dI::AbstractVector{<:Number}
end

function Base.show(io::IO, layer::Layer)
    print("  $(length(layer.activation_fns)) neurons\n")
    print("  Weights " *
          "[$(size(layer.weights, 1))x$(size(layer.weights, 2))]: " *
          "$(layer.weights)\n")
end

function Base.show(io::IO, nn::NeuralNetwork)
    num_layers = length(nn.layers)
    print("Neural network with $(num_layers) layers.\n")
    print("Input dimensionality: $(in_ndims_sans_bias(nn.layers[1]))\n")
    print("Output dimensionality: $(out_ndims(nn.layers[num_layers]))\n")
    for l in 1:num_layers
        print("Layer $(l):\n")
        print(nn.layers[l])
    end
end

function Base.show(io::IO, layer_forward_pass::LayerForwardPass)
    print("Input vector: $(layer_forward_pass.inputs), " *
          "$(size(layer_forward_pass.inputs))\n")
    print("Intermediate sums: $(layer_forward_pass.sums), " *
          "$(size(layer_forward_pass.sums))\n")
    print("Output vector: $(layer_forward_pass.activations), " *
          "$(size(layer_forward_pass.activations))\n")
end

function Base.show(io::IO, layer_backward_pass::LayerBackwardPass)
    print("dL_dS: $(layer_backward_pass.dL_dS), " *
          "$(size(layer_backward_pass.dL_dS))\n")
    print("dL_dW: $(layer_backward_pass.dL_dW), " *
          "$(size(layer_backward_pass.dL_dW))\n")
    print("dL_dI: $(layer_backward_pass.dL_dI), " *
          "$(size(layer_backward_pass.dL_dI))\n")
end

"""
Data structure containing loss stats for each epoch of training.
"""
struct TrainingStats
    losses::AbstractVector
end

##
## Constructors
##

"""
    create_layer(in_ndims_sans_bias, out_ndims, activation_fns)

Creates a single layer of a neural network with the given dimensions and
activation functions, and randomly assigned weights.

In this package, a "layer" is composed of (1) weights, which are applied to the
previous layer, and (2) activation functions, which are applied after weights.
"""
function create_layer(in_ndims_sans_bias::Integer,
                      out_ndims::Integer,
                      activation_fns::AbstractVector{<:Function})::Layer
    @assert length(activation_fns) == out_ndims
    weights = randn((in_ndims_sans_bias+1, out_ndims))
    return Layer(weights, activation_fns)
end

"""
    validate_layer(layer, num_inputs_sans_bias, num_outputs)

Asserts that the layer parameters have the desired dimensionality, not including
the bias.
"""
function validate_layer(layer::Layer,
                        num_inputs_sans_bias::Integer,
                        num_outputs::Integer)
    @assert in_ndims_sans_bias(layer) == num_inputs_sans_bias
    @assert out_ndims(layer) == num_outputs
end

"""
    compose_layers(input_ndims, output_ndims, loss_fn, layers...)

Creates a neural network from the supplied layers.
"""
function compose_layers(input_ndims::Integer,
                        output_ndims::Integer,
                        loss_fn::Function,
                        layers...)::NeuralNetwork
    num_layers = length(layers)
    nn_layers = Vector{Layer}(undef, num_layers)

    # Verify layers are dimensionally compatible.
    for l in 1:num_layers
        expected_in_ndims_sans_bias = (l == 1 ?
                                       input_ndims :
                                       out_ndims(layers[l-1]))
        expected_out_ndims = (l == num_layers ?
                              output_ndims :
                              in_ndims_sans_bias(layers[l+1]))
        validate_layer(layers[l],
                       expected_in_ndims_sans_bias,
                       expected_out_ndims)
    end

    return NeuralNetwork([layers...], loss_fn)
end

"""
    create_nn(layer_dims, activation_fn, loss_fn,
              network_input_dims, network_output_dims)

Creates a neural network by building layers with the specified dimensionality,
`activation_fn`, and `loss_fn`.

- `layer_dims::AbstractVector`: a vector of integers, where the `n`-th value
corresponds to the number of neurons in the `n`-th layer of the network.
- `activation_fn`: the activation function used for all nodes in the network.
- `loss_fn`: a single loss function applied at the end of a network, producing a
  scalar loss value.
- `network_input_dims`: the dimensionality of inputs to the network.
- `network_output_dims`: the dimensionality of outputs of the network.

TODO: This function is convenient, but it's probably better to construct and
compose layers individually.
"""
function create_nn(layer_dims::AbstractVector{<:Integer},
                   activation_fn::Function,
                   loss_fn::Function,
                   network_input_ndims::Integer,
                   network_output_ndims::Integer)::NeuralNetwork
    num_layers = length(layer_dims)
    layers = Vector{Layer}(undef, num_layers)
    for l in 1:num_layers
        in_ndims_sans_bias = (l == 1 ?
                              network_input_ndims :
                              layer_dims[l-1])
        out_ndims = layer_dims[l]
        layers[l] = create_layer(in_ndims_sans_bias, out_ndims,
                                 fill(activation_fn, out_ndims))
    end

    return compose_layers(network_input_ndims, network_output_ndims,
                          loss_fn, layers...)
end

##
## Training and inference
##

"""
    run_forward_pass(layer, inputs)

Computes the activations of `layer` on `inputs`, returning the cached results.
"""
function run_forward_pass(layer::Layer,
                          inputs::AbstractVector{<:Number})::LayerForwardPass
    @assert ndims(inputs) == 1
    @assert length(inputs) == in_ndims_sans_bias(layer)

    # Append the bias term.
    inputs = copy(inputs)
    push!(inputs, 1)
    # Perform matrix multiplication `weights`^T * `input`.
    # For the `weights` matrix, the rows index the inputs and the columns index
    # the outputs. `weights` is transposed before multiplication so that the
    # rows index outputs, and the result is a column vector with an entry for
    # each neuron output in this layer.
    sums = transpose(layer.weights) * inputs
    # Apply each neuron's activation function.
    activations = map.(layer.activation_fns, sums)

    return LayerForwardPass(inputs, sums, activations)
end

"""
    compute_loss_gradient(net_output, label, loss_fn)

Computes the gradient of `loss_fn` parameterized by `label`, with respect to
`net_output`, evaluated at `net_output`.

- `loss_fn`: a function f: (R^n * R^n) -> R
"""
function compute_loss_gradient(net_output::AbstractVector{<:Number},
                               label::AbstractVector{<:Number},
                               loss_fn::Function)::AbstractVector{<:Number}
    @assert length(net_output) == length(label)
    
    partial_loss(x) = loss_fn(label, x)
    dL_dO = gradient(partial_loss, net_output)[1]

    return dL_dO
end

"""
    run_backward_pass(layer, forward_pass, dL_dO)

Does backprop to compute the loss gradients with respect to the components of a
single layer, using the values from the `forward_pass`.

Returns a LayerBackwardPass containing the gradients, without mutating the
original layer.
"""
function run_backward_pass(layer::Layer,
                           forward_pass::LayerForwardPass,
                           dL_dO::AbstractVector{<:Number})::LayerBackwardPass
    @assert length(dL_dO) == out_ndims(forward_pass)

    # dL_dS|S = dL_dO|O * dO_dS|S
    # dO_dS is just the derivative of the activation function.
    dO_dS = map(gradient, layer.activation_fns, forward_pass.sums) |>
        vec -> map(t -> t[1], vec)
    dL_dS = dL_dO .* dO_dS
    # dL_dW|W = dL_dS|S * dS_dW|W
    # dS_dW for each weight is the value of the output which is scaled by that
    # weight.
    dL_dW = forward_pass.inputs * transpose(dL_dS)

    # dL_dI|I = sum(dL_dS|S * dS_dI|I)
    # dS_dI is just the applied weight; the matrix multiplication applies the
    # vectorized sum.
    dL_dI = layer.weights * dL_dS

    # Note that the last element in dL_dW corresponds to bias, and so should be
    # omitted by the previous layer during backprop.
    return LayerBackwardPass(dL_dS, dL_dW, dL_dI)
end

"""
    update_layer_weights!(layer, backward_pass, learning_rate)

Updates weights according to backprop gradient calculation by applying values
from a previously computed `backward_pass`.
"""
function update_layer_weights!(layer::Layer,
                               backward_pass::LayerBackwardPass,
                               learning_rate::Number)
    @assert size(layer.weights) == size(backward_pass.dL_dW)

    layer.weights .-= (learning_rate .* backward_pass.dL_dW)
end

"""
    train!(nn, samples, labels, learning_rate)
"""
function train!(nn::NeuralNetwork,
                samples::AbstractVector,
                labels::AbstractVector,
                learning_rate::Number)::TrainingStats
    @assert length(samples) == length(labels)

    num_layers = length(nn.layers)

    # Record the results of forward and backward passes for each layer.
    # These are only used for backprop and are overwritten on every epoch.
    forward_passes = Vector{LayerForwardPass}(undef, num_layers)
    backward_passes = Vector{LayerBackwardPass}(undef, num_layers)

    losses = Vector(undef, length(samples))

    for (index, input) in enumerate(samples)
        label = labels[index]

        # Run forward-pass.
        for l in 1:num_layers
            forward_passes[l] = run_forward_pass(nn.layers[l], input)
            input = forward_passes[l].activations
        end

        # Compute loss.
        loss = nn.loss_fn(input, label)

        # Backprop.
        loss_grad = compute_loss_gradient(input, label, nn.loss_fn)

        for l in reverse(1:num_layers)
            backward_passes[l] =
                run_backward_pass(nn.layers[l],
                                  forward_passes[l],
                                  l == num_layers ?
                                  loss_grad :
                                  backward_passes[l+1].dL_dI[1:end-1])
        end

        # Update weights. The weight updates have already been computed, so this
        # can be done in any order.
        for l in 1:num_layers
            update_layer_weights!(nn.layers[l],
                                  backward_passes[l],
                                  learning_rate)
        end

        losses[index] = loss
        if (index % 1000 == 0)
            print("Iteration $index / $(length(samples)) : loss=$(loss)\n")
        end
    end

    return TrainingStats(losses)
end

"""
    predict(nn, inputs)

Computes the end-to-end application of `nn` on `inputs`.
"""
function predict(nn::NeuralNetwork, inputs::AbstractVector{<:Number})
    for layer in nn.layers
        inputs = run_forward_pass(layer, inputs).activations
    end
    return inputs
end

##
## Vectorized implementations (experimental)
##

# It's not ideal that the implementation is forked, but it's rather challenge to
# have a single implementation that handles both vectorized and non-vectorized
# updates. In particular, the vectorized code is significantly more complex, so
# most parts are minimally commented, as they assume prior understanding of the
# underlying math.

"""
Vectorized forward-pass data structure.
"""
struct LayerForwardPassVectorized
    # (n * d_in+1)
    inputs::AbstractMatrix{<:Number}
    # (n * d_out)
    sums::AbstractMatrix{<:Number}
    # (n * d_out)
    activations::AbstractMatrix{<:Number}
end

"""
Vectorized backward-pass data structure.
"""
struct LayerBackwardPassVectorized
    # (n * d_out)
    dL_dS::AbstractMatrix{<:Number}
    # (n * d_in+1 * d_out)
    dL_dW::AbstractArray{<:Number, 3}
    # (n * d_in)
    dL_dI::AbstractMatrix{<:Number}
end

"""
    run_forward_pass_vectorized(layer, inputs)

Vectorized forward-pass.

- `inputs`: (n * d_in) matrix of samples.
"""
function run_forward_pass_vectorized(layer::Layer,
                                     inputs::AbstractMatrix{<:Number})::LayerForwardPassVectorized
    @assert ndims(inputs) == 2
    @assert size(inputs, 2) == in_ndims_sans_bias(layer)

    num_samples = size(inputs, 1)
    # Append the bias column (extra dimension for each sample).
    biased_inputs = [inputs ones(num_samples,1)]
    # (n * d_in+1) * (d_in+1 * d_out)
    sums = biased_inputs * layer.weights

    # Magic!
    # `mapslices(f, m, dims=2)` broadcasts f to each row of m.
    # `map.(f, v)` performs elementwise function application for equal sized
    # vectors of functoins and elements.
    # The net result is that the dth function is applied to every element in the
    # dth column.
    activations = mapslices(row -> map.(layer.activation_fns, row), sums, dims=2)

    return LayerForwardPassVectorized(biased_inputs, sums, activations)
end

"""
    compute_loss_gradient_vectorized(net_outputs, labels, loss_fn)

Vectorized gradient computation.

- `loss_fn`: a function that computes a scalar loss from two matrices.
- `net_outputs`: (n * d) matrix of output predictions (rows are samples).
- `labels`: (n * d) matrix of labels (rows are samples).

Returns (n * d) matrix of gradients.
"""
function compute_loss_gradient_vectorized(net_outputs::AbstractMatrix{<:Number},
                                          labels::AbstractMatrix{<:Number},
                                          loss_fn::Function)::AbstractMatrix{<:Number}
    @assert size(net_outputs) == size(labels)

    partial_loss(xs) = loss_fn(labels, xs)
    dL_dO = gradient(partial_loss, net_outputs)[1]

    return dL_dO
end

"""
    run_backward_pass_vectorized(layer, forward_pass, dL_dO)

Vectorized backward-pass.

- `dL_dO`: (n * d_out) matrix of gradients.
"""
function run_backward_pass_vectorized(layer::Layer,
                                      forward_pass::LayerForwardPassVectorized,
                                      dL_dO::AbstractMatrix{<:Number})::LayerBackwardPassVectorized
    @assert size(dL_dO, 2) == out_ndims(forward_pass)

    num_samples = size(dL_dO, 1)

    # Slice by row, and for each, do element-wise activation function
    # application.
    # forward_pass.sums is (n * d_out).
    # (n * d_out)
    dO_dS = mapslices(row -> map(gradient, layer.activation_fns, row),
                      forward_pass.sums,
                      dims=[2]) |>
        vec -> map(t -> t[1], vec)
    # (n * d_out)
    dL_dS = dL_dO .* dO_dS

    # Slice by row (each row is a (d_in+1 * d_out) matrix!), and for each
    # compute the outer product which computes the weight gradients for the
    # given sample.
    # inputs are (n * d_in+1), dL_dS is (n * d_out).
    # (n * d_in+1 * d_out)
    dL_dW = permutedims(
        cat([forward_pass.inputs[i,:] * transpose(dL_dS[i,:]) for i in num_samples]...,
            dims=3),
        (3,1,2))

    # Weights are (d_in+1, d_out)
    # (n * d_in+1)
    dL_dI = dL_dS * transpose(layer.weights)

    return LayerBackwardPassVectorized(dL_dS, dL_dW, dL_dI)
end

"""
    update_layer_weights_vectorized!(layer, backward_pass, learning_rate)

Vectorized layer weight update
"""
function update_layer_weights_vectorized!(layer::Layer,
                                          backward_pass::LayerBackwardPassVectorized,
                                          learning_rate::Number)
    @assert size(layer.weights) == size(backward_pass.dL_dW)[2:end]

    layer.weights .-= (learning_rate .*
                       dropdims(sum(backward_pass.dL_dW, dims=1), dims=1))
end

"""
    train_vectorized!(nn, samples, labels, learning_rate)

Vectorized gradient descent.
"""
function train_vectorized!(nn::NeuralNetwork,
                           samples::AbstractVector,
                           labels::AbstractVector,
                           learning_rate::Number,
                           batch_size::Integer)::TrainingStats
    @assert length(samples) == length(labels)

    num_layers = length(nn.layers)

    forward_passes = Vector{LayerForwardPassVectorized}(undef, num_layers)
    backward_passes = Vector{LayerBackwardPassVectorized}(undef, num_layers)

    # Divide the data into batch_size chunks.
    sample_batches = Iterators.partition(samples, batch_size) |> collect
    label_batches = Iterators.partition(labels, batch_size) |> collect

    num_passes = length(sample_batches)
    losses = Vector(undef, num_passes)

    for (index, input) in enumerate(sample_batches)
        input_batch = transpose(hcat(input...))
        label_batch = transpose(hcat(label_batches[index]...))

        # Run forward-pass.
        for l in 1:num_layers
            forward_passes[l] = run_forward_pass_vectorized(nn.layers[l], input_batch)
            input_batch = forward_passes[l].activations
        end

        # Compute loss.
        loss = nn.loss_fn(input_batch, label_batch)

        # Backprop.
        loss_grad = compute_loss_gradient_vectorized(input_batch,
                                                     label_batch,
                                                     nn.loss_fn)

        for l in reverse(1:num_layers)
            backward_passes[l] =
                run_backward_pass_vectorized(nn.layers[l],
                                             forward_passes[l],
                                             l == num_layers ?
                                             loss_grad :
                                             # Chop off the bias column.
                                             backward_passes[l+1].dL_dI[:,1:end-1])
        end

        # Update weights. The weight updates have already been computed, so this
        # can be done in any order.
        for l in 1:num_layers
            update_layer_weights_vectorized!(nn.layers[l],
                                             backward_passes[l],
                                             learning_rate)
        end

        losses[index] = loss
        if (index % 1000 == 0)
            print("Iteration $index / $(num_passes) : loss=$(loss)\n")
        end
    end

    return TrainingStats(losses)
end

##
## Miscellaneous
##

"""
    in_ndims_sans_bias(layer)

Returns the number of inputs to `layer`, not including the bias term (in other
words, only the inputs that come from the previous layer).
"""
function in_ndims_sans_bias(layer::Layer)
    return size(layer.weights, 1) - 1
end

"""
    in_ndims_sans_bias(layer)

Returns the number of inputs to `layer`, not including the bias term (in other
words, only the inputs that come from the previous layer).
"""
function in_ndims_sans_bias(layer::LayerBackwardPass)
    @assert length(dL_dI) == size(layer.dL_dW, 1) - 1
    return length(dL_dI)
end

"""
    in_ndims_with_bias(layer)

Returns the number of inputs to `layer`, including the bias term.
"""
function in_ndims_with_bias(layer::Layer)
    return size(layer.weights, 1)
end

"""
    out_ndims(layer)

Returns the number of outputs from `layer`.
"""
function out_ndims(layer::Layer)
    @assert size(layer.weights, 2) == length(layer.activation_fns)
    return size(layer.weights, 2)
end

"""
    out_ndims(layer)

Returns the number of outputs from `layer`.
"""
function out_ndims(layer::LayerForwardPass)
    @assert length(layer.sums) == length(layer.activations)
    return length(layer.activations)
end

"""
    out_ndims(layer)

Returns the number of outputs from `layer`.
"""
function out_ndims(layer::LayerForwardPassVectorized)
    @assert size(layer.sums) == size(layer.activations)

    return size(layer.activations, 2)
end

"""
DEPRECATED

    compute_loss(net_output, label, loss_fn)

Computes the loss between `label` and `net_output` using `loss_fn`. Despite the
loss being a scalar, this transforms it into a singleton array for consistency
with the multi-objective function.

- `loss_fn`: a function f: (R^n * R^n) -> R
"""
function compute_loss(net_output::AbstractVector{<:Number},
                      label::AbstractVector{<:Number},
                      loss_fn::Function)::AbstractVector{<:Number}
    return [loss_fn(label, net_output)]
end

"""
DEPRECATED

    compute_multi_objective_loss_gradient(net_output, label, loss_fn)

Computes the gradient of `loss_fn` parameterized by `label`, with respect to
`net_output`, evaluated at `net_output`.

You probably don't want to do this (this package does not provide a good
solution to multi-objective optimization problems).

- `loss_fn`: a function f: (R^n * R^n) -> R^n
"""
function compute_multi_objective_loss_gradient(net_output::AbstractVector{<:Number},
                                               label::AbstractVector{<:Number},
                                               loss_fn::Function)::AbstractVector{<:Number}
    @assert length(net_output) == length(label)
 
    # Vector of partially-applied loss functions, where each entry lossmap[i]
    # contains f(x) = loss_fn(label[i], x)
    lossmap = map(y -> (x -> loss_fn(y, x)), label)

    # Take the gradient of each loss function with respect to each output,
    # evaluated at the outputs.
    # dL_dO is just the derivative of the loss function.
    dL_dO = map(gradient, lossmap, net_output) |>
        vec -> map(t -> t[1], vec)

    return dL_dO
end

"""
DEPRECATED

    compute_multi_objective_loss(net_output, label, loss_fn)

Computes the loss between `label` and `net_output` using `loss_fn`.

You probably don't want to do this (this package does not provide a good
solution to multi-objective optimization problems).

- `loss_fn`: a function f: (R^n * R^n) -> R^n
"""
function compute_multi_objective_loss(net_output::AbstractVector{<:Number},
                                      label::AbstractVector{<:Number},
                                      loss_fn::Function)::AbstractVector{<:Number}
    # Vector of partially-applied loss functions, where each entry lossmap[i]
    # contains f(x) = loss_fn(label[i], x)
    lossmap = map(y -> (x -> loss_fn(y, x)), label)

    # Returns a vector loss.
    return map.(lossmap, net_output)
end

end # module
