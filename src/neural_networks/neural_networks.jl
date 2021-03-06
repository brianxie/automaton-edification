module NeuralNetworks

# TODO: Add exports

using LinearAlgebra, Zygote, ForwardDiff

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
    # Vector-valued function (f: R^d -> R^d) mapping the neurons in this layer
    # to their activations.
    activation_fn::Function
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
    print("  $(size(layer.weights, 2)) neurons\n")
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
    create_layer(in_ndims_sans_bias, out_ndims, activation_fn)

Creates a single layer of a neural network with the given dimensions and
activation function, and randomly assigned weights.

In this package, a "layer" is composed of (1) weights, which are applied to the
previous layer, and (2) activation functions, which are applied after weights.
"""
function create_layer(in_ndims_sans_bias::Integer,
                      out_ndims::Integer,
                      activation_fn::Function)::Layer
    weights = randn((in_ndims_sans_bias+1, out_ndims)) ./ sqrt(in_ndims_sans_bias)
    return Layer(weights, activation_fn)
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
- `activation_fn`: the activation function used for all layers in the network.
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
                                 activation_fn)
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

    # Apply each neuron's activation function. The activation function is
    # vector-valued, so the result is a vector of the same dimensionality.
    activations = layer.activation_fn(sums)

    return LayerForwardPass(inputs, sums, activations)
end

"""
    compute_loss_gradient(net_output, label, loss_fn)

Computes the gradient of `loss_fn` parameterized by `label`, with respect to
`net_output`, evaluated at `net_output`.

- `loss_fn`: a function f: (R^n * R^n) -> R

Returns a vector with the size of the output layer.
"""
function compute_loss_gradient(net_output::AbstractVector{<:Number},
                               label::AbstractVector{<:Number},
                               loss_fn::Function)::AbstractVector{<:Number}
    @assert length(net_output) == length(label)
    
    partial_loss(x) = loss_fn(x, label)
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
    # dO_dS is just the derivative (Jacobian) of the activation function.
    # (d_out * d_out)
    dO_dS = ForwardDiff.jacobian(layer.activation_fn, forward_pass.sums)

    # dL_dS = dL_dO * dO_dS
    # For activation functions which are elementwise-only, this is actually
    # equivalent to a vector-vector Hadamard product, because the Jacobian is
    # diagonal.
    # (d_out)
    dL_dS = transpose(dO_dS) * dL_dO

    # dL_dW|W = dL_dS|S * dS_dW|W
    # dS_dW for each weight is the value of the output which is scaled by that
    # weight.
    # (d_in+1 * d_out)
    dL_dW = forward_pass.inputs * transpose(dL_dS)

    # dL_dI|I = sum(dL_dS|S * dS_dI|I)
    # dS_dI is just the applied weight; the matrix multiplication applies the
    # vectorized sum.
    # (d_in+1)
    dL_dI = layer.weights * dL_dS

    # Note that the last element in dL_dW corresponds to bias, and so should be
    # omitted by the previous layer during backprop.
    return LayerBackwardPass(dL_dS, dL_dW, dL_dI)
end

"""
    update_layer_weights!(layer, backward_pass, num_samples, learning_rate)

Updates weights according to backprop gradient calculation by applying values
from a previously computed `backward_pass`.
"""
function update_layer_weights!(layer::Layer,
                               backward_pass::LayerBackwardPass,
                               num_samples::Number,
                               learning_rate::Number)
    @assert size(layer.weights) == size(backward_pass.dL_dW)

    layer.weights .-= (learning_rate .* backward_pass.dL_dW ./ num_samples)
end

"""
    train!(nn, samples, labels, learning_rate, batch_size)
"""
function train!(nn::NeuralNetwork,
                samples::AbstractVector,
                labels::AbstractVector,
                learning_rate::Number,
                batch_size::Integer)::TrainingStats
    @assert length(samples) == length(labels)

    num_layers = length(nn.layers)

    # Divide the data into batch_size chunks.
    sample_batches = Iterators.partition(samples, batch_size) |> collect
    label_batches = Iterators.partition(labels, batch_size) |> collect

    num_passes = length(sample_batches)

    losses = Vector(undef, length(sample_batches))

    for (batch_index, input_batch) in enumerate(sample_batches)
        label_batch = label_batches[batch_index]
        num_samples = length(input_batch)

        # Record the results of forward and backward passes for each layer.
        # These are only used for backprop and are overwritten on every batch.
        # Rows index samples; columns index layers.
        forward_passes = Matrix{LayerForwardPass}(undef, (num_samples, num_layers))
        backward_passes = Matrix{LayerBackwardPass}(undef, (num_samples, num_layers))

        batch_loss = 0.0

        # Compute updates for the batch.
        for (point_index, input_point) in enumerate(input_batch)
            point_label = label_batch[point_index]

            # Run forward-pass.
            for l in 1:num_layers
                forward_passes[point_index, l] = run_forward_pass(nn.layers[l], input_point)
                input_point = forward_passes[point_index, l].activations
            end

            # Compute loss.
            batch_loss += nn.loss_fn(input_point, point_label)

            # Backprop.
            loss_grad = compute_loss_gradient(input_point, point_label, nn.loss_fn)

            for l in reverse(1:num_layers)
                backward_passes[point_index, l] =
                    run_backward_pass(nn.layers[l],
                                      forward_passes[point_index, l],
                                      l == num_layers ?
                                      loss_grad :
                                      backward_passes[point_index, l+1].dL_dI[1:end-1])
            end
        end

        # Update weights. The weight updates have already been computed, so this
        # can be done in any order.
        for i in 1:num_samples, l in 1:num_layers
            update_layer_weights!(nn.layers[l],
                                  backward_passes[i, l],
                                  num_samples,
                                  learning_rate)
        end

        batch_loss /= num_samples

        losses[batch_index] = batch_loss
        if (batch_index % 1000 == 0)
            print("Iteration $batch_index / $(length(sample_batches)) : " *
                  "loss=$(batch_loss)\n")
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
## Vectorized implementations
##

# It's not ideal that the implementation is forked, but it's rather challenging
# to have a single implementation that handles both vectorized and
# non-vectorized updates. In particular, the vectorized code is significantly
# more complex, so is not optimized for clarity.

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
    # Allocate space for the Jacobian of the activation function. The partial
    # derivatives can't be hard-coded since activation functions may vary, and
    # each sample requires a (d * d) matrix, since the activation function can
    # map a vector to a vector.
    # (n * d_out * d_out)
    dO_dS::AbstractArray{<:Number, 3}
    # (n * d_out)
    dL_dS::AbstractMatrix{<:Number}
    # Since only the total gradient update is needed, the n-dimension is collapsed.
    # Otherwise, this would be a 3D (n * d_in+1 + d_out) array.
    # (d_in+1 * d_out)
    dL_dW::AbstractMatrix{<:Number}
    # (n * d_in+1)
    dL_dI::AbstractMatrix{<:Number}
end

"""
    run_forward_pass_vectorized!(layer, inputs, num_samples, batch_size,
                                 forward_pass_out)

Vectorized forward-pass.

Content of extra rows when `num_samples` < `batch_size` is undefined.

- `inputs`: (n * d_in) matrix of samples.
- `num_samples`: n_s, number of input samples.
- `batch_size`: n, batch size, corresponding to the number of rows preallocated
  for various matrices; may be larger than `num_samples` for incomplete batches.
"""
function run_forward_pass_vectorized!(layer::Layer,
                                      inputs::AbstractMatrix{<:Number},
                                      num_samples::Integer,
                                      batch_size::Integer,
                                      forward_pass_out::LayerForwardPassVectorized)
    @assert ndims(inputs) == 2
    @assert size(inputs, 2) == in_ndims_sans_bias(layer)

    # Append the bias column (extra dimension for each sample).
    forward_pass_out.inputs .= [inputs ones(batch_size,1)]

    # sums = biased_inputs * layer.weights
    # (n * d_in+1) * (d_in+1 * d_out)
    mul!(forward_pass_out.sums, forward_pass_out.inputs, layer.weights)

    # 1/ One-liner alternative for multiple activation functions:
    #
    # activations = mapslices(row -> map.(layer.activation_fns, row), sums, dims=2)
    #
    # `mapslices(f, m, dims=2)` broadcasts f to each row of m.
    # `map.(f, v)` performs elementwise function application for equal sized
    # vectors of functions and elements.
    # The net result is that the dth function is applied to every element in the
    # dth column.
    #
    # 2/ Straightforward, but optimized, version:
    #
    # Threads.@threads for j in 1:out_ndims(layer)
    #     map!(layer.activation_fns[j],
    #          view(forward_pass_out.activations, 1:num_samples, j),
    #          view(forward_pass_out.sums, 1:num_samples, j))
    # end

    # Skip activation computation for the last (batch_size - num_samples) rows.
    # Poor cache access pattern.
    Threads.@threads for i in 1:num_samples
        view(forward_pass_out.activations, i, :) .=
            layer.activation_fn(view(forward_pass_out.sums, i, :))
    end
end

"""
    compute_loss_gradient_vectorized!(net_outputs, labels, loss_fn,
                                      num_samples, batch_size,
                                      loss_grad_out)

Vectorized gradient computation.

Content of extra rows when n_s < n is undefined.

- `net_outputs`: (n * d) matrix of output predictions (rows are samples).
- `labels`: (n_s * d) matrix of labels (rows are samples); note that there may
  be a size mismatch between n and n_s if the batch is incomplete.
- `loss_fn`: a function that computes a scalar loss from two matrices.
- `num_samples`: n_s, number of input samples.
- `batch_size`: n, batch size, corresponding to the number of rows preallocated
  for various matrices; may be larger than `num_samples` for incomplete batches.

Returns (n * d) matrix of gradients.

This function may return a different (scaled) result compared to
`compute_loss_gradient`. When the values disagree, the non-vectorized version
should be treated as the reference implementation.
"""
function compute_loss_gradient_vectorized!(net_outputs::AbstractMatrix{<:Number},
                                           labels::AbstractMatrix{<:Number},
                                           loss_fn::Function,
                                           num_samples::Integer,
                                           batch_size::Integer,
                                           loss_grad_out::AbstractMatrix{<:Number})
    @assert size(net_outputs, 2) == size(labels, 2)

    partial_loss(xs) = loss_fn(xs, labels)

    # Compute dL_dO.
    if num_samples == batch_size
        loss_grad_out .= gradient(partial_loss, net_outputs)[1]
    else
        view(loss_grad_out, 1:num_samples, :) .=
            gradient(partial_loss, view(net_outputs, 1:num_samples, :))[1]
    end
end

"""
    run_backward_pass_vectorized!(layer, forward_pass, dL_dO, num_samples, batch_size,
                                  backward_pass_out)

Vectorized backward-pass.

Content of extra rows when `num_samples` < `batch_size` is undefined.

- `dL_dO`: (n * d_out) matrix of gradients.
- `num_samples`: n_s, number of input samples.
- `batch_size`: n, batch size, corresponding to the number of rows preallocated
  for various matrices; may be larger than `num_samples` for incomplete batches.
"""
function run_backward_pass_vectorized!(layer::Layer,
                                       forward_pass::LayerForwardPassVectorized,
                                       dL_dO::AbstractMatrix{<:Number},
                                       num_samples::Integer,
                                       batch_size::Integer,
                                       backward_pass_out::LayerBackwardPassVectorized)
    @assert size(dL_dO, 2) == out_ndims(forward_pass)

    # 1/ One-liner alternative for multiple activation functions:
    #
    # Slice by row, and for each, do element-wise activation function
    # application.
    #
    # dO_dS = mapslices(row -> map(gradient, layer.activation_fns, row),
    #                   forward_pass.sums,
    #                   dims=2) |>
    #     vec -> map(t -> t[1], vec)
    #
    # 2/ Optimized version:
    #
    # Threads.@threads for j in 1:length(layer.activation_fns)
    #     for i in 1:num_samples
    #         backward_pass_out.dL_dS[i,j] = gradient(layer.activation_fns[j],
    #                                                 forward_pass.sums[i,j])[1]
    #     end
    # end

    # Skip the last (batch_size - num_samples) rows.
    # Poor cache access pattern.
    # (n * d_out * d_out)
    Threads.@threads for i in 1:num_samples
        ForwardDiff.jacobian!(view(backward_pass_out.dO_dS, i, :, :),
                              layer.activation_fn,
                              view(forward_pass.sums, i, :))
    end

    # dL_dS = dL_dO * dO_dS
    # (n * d_out)
    # Poor cache access pattern.
    Threads.@threads for i in 1:num_samples
        mul!(view(backward_pass_out.dL_dS, i, :),
             transpose(view(backward_pass_out.dO_dS, i, :, :)),
             view(dL_dO, i, :))
    end

    # 1/ One-liner for 3D solution (keep each sample, (n * d_in+1 * d_out))
    # Slice by row (each row is a (d_in+1 * d_out) matrix!), and for each
    # compute the outer product which computes the weight gradients for the
    # given sample.
    #
    # dL_dW = permutedims(
    #     cat([forward_pass.inputs[i,:] * transpose(dL_dS[i,:]) for i in 1:num_samples]...,
    #         dims=3),
    #     (3,1,2))

    # 2/ Vectorized 3D, (n * d_in * d_out)
    #
    # dL_dW = Array{Float64, 3}(undef, (num_samples, # n
    #                                   size(forward_pass.inputs, 2), # d_in+1
    #                                   size(dL_dS, 2))) # d_out
    # for i in 1:num_samples
    #     # Poor cache locality
    #     dL_dW[i,:,:] .= forward_pass.inputs[i,:] * transpose(dL_dS[i,:])
    # end

    # inputs are (n * d_in+1), dL_dS is (n * d_out).
    # (d_in * d_out)
    fill!(backward_pass_out.dL_dW, 0.0)
    # TODO: Parallelize this with thread-local memory, and combine at end
    # Skip the last (batch_size - num_samples) rows.
    for i in 1:num_samples
        # dL_dW .+= forward_pass.inputs[i,:] * transpose(dL_dS[i,:])
        BLAS.axpy!(1,
                   view(forward_pass.inputs,i,:) *
                   transpose(view(backward_pass_out.dL_dS,i,:)),
                   backward_pass_out.dL_dW)
    end

    # Weights are (d_in+1, d_out)
    # (n * d_in+1)
    # dL_dI = dL_dS * transpose(layer.weights)
    # Uses BLAS to avoid allocation for the transpose.
    BLAS.gemm!('N',
               'T',
               1.0,
               backward_pass_out.dL_dS,
               layer.weights,
               0.0,
               backward_pass_out.dL_dI)
end

"""
    update_layer_weights_vectorized!(layer, backward_pass, learning_rate)

Vectorized layer weight update
"""
function update_layer_weights_vectorized!(layer::Layer,
                                          backward_pass::LayerBackwardPassVectorized,
                                          learning_rate::Number)
    @assert size(layer.weights) == size(backward_pass.dL_dW)

    # One-liner for n-dimension collapse:
    # layer.weights .-= (learning_rate .*
    #                    dropdims(sum(backward_pass.dL_dW, dims=1), dims=1))

    BLAS.axpy!(-learning_rate, backward_pass.dL_dW, layer.weights)
end

"""
    train_vectorized!(nn, samples, labels, learning_rate, batch_size)

Vectorized gradient descent.

TODO: The conditional logic involved in properly handling partial batches causes
a noticeable performance regression.
"""
function train_vectorized!(nn::NeuralNetwork,
                           samples::AbstractVector,
                           labels::AbstractVector,
                           learning_rate::Number,
                           batch_size::Integer)::TrainingStats
    @assert length(samples) == length(labels)

    num_layers = length(nn.layers)

    # Divide the data into batch_size chunks.
    sample_batches = Iterators.partition(samples, batch_size) |>
        batches -> map(batch -> transpose(hcat(batch...)), batches) |>
        collect
    label_batches = Iterators.partition(labels, batch_size) |>
        batches -> map(batch -> transpose(hcat(batch...)), batches) |>
        collect

    num_passes = length(sample_batches)

    # Preallocate memory for each batch iteration.
    forward_passes = Vector{LayerForwardPassVectorized}(undef, num_layers)
    for l in 1:num_layers
        inputs = zeros(batch_size, in_ndims_with_bias(nn.layers[l]))
        sums = zeros(batch_size, out_ndims(nn.layers[l]))
        activations = zeros(batch_size, out_ndims(nn.layers[l]))
        forward_passes[l] = LayerForwardPassVectorized(inputs, sums, activations)
    end

    backward_passes = Vector{LayerBackwardPassVectorized}(undef, num_layers)
    for l in 1:num_layers
        dO_dS = zeros(batch_size, out_ndims(nn.layers[l]), out_ndims(nn.layers[l]))
        dL_dS = zeros(batch_size, out_ndims(nn.layers[l]))
        dL_dW = zeros(in_ndims_with_bias(nn.layers[l]),
                      out_ndims(nn.layers[l]))
        dL_dI = zeros(batch_size, in_ndims_with_bias(nn.layers[l]))
        backward_passes[l] = LayerBackwardPassVectorized(dO_dS, dL_dS, dL_dW, dL_dI)
    end

    loss_grad = zeros(batch_size, out_ndims(nn.layers[end]))

    losses = Vector(undef, num_passes)

    for (index, input_batch) in enumerate(sample_batches)
        num_samples = size(input_batch, 1)

        # Individual subroutines expect matrices with size batch_size, but
        # otherwise correctly handle mismatches between num_samples and
        # batch_size.
        if (num_samples != batch_size)
            print("Incomplete batch; $num_samples / $batch_size\n")
            input_batch = vcat(input_batch,
                               zeros(batch_size - num_samples,
                                     size(input_batch, 2)))
        end

        # Run forward-pass.
        for l in 1:num_layers
            run_forward_pass_vectorized!(nn.layers[l],
                                         l == 1 ?
                                             input_batch :
                                             forward_passes[l-1].activations,
                                         num_samples,
                                         batch_size,
                                         forward_passes[l])
        end

        # Compute loss.
        # Avoid taking a view unless there isn't a full batch.
        loss = (num_samples == batch_size) ?
            nn.loss_fn(forward_passes[num_layers].activations,
                       label_batches[index]) :
            nn.loss_fn(
                view(forward_passes[num_layers].activations,1:num_samples),
                label_batches[index])

        # Backprop.
        compute_loss_gradient_vectorized!(forward_passes[num_layers].activations,
                                          label_batches[index],
                                          nn.loss_fn,
                                          num_samples,
                                          batch_size,
                                          loss_grad)

        for l in reverse(1:num_layers)
            run_backward_pass_vectorized!(nn.layers[l],
                                          forward_passes[l],
                                          l == num_layers ?
                                          loss_grad :
                                          # Chop off the bias column.
                                          view(backward_passes[l+1].dL_dI,
                                               :, 1:out_ndims(nn.layers[l])),
                                          num_samples,
                                          batch_size,
                                          backward_passes[l])
        end

        # Update weights. The weight updates have already been computed, so this
        # can be done in any order.
        # This could be multithreaded, but the network would need to be very
        # deep for the parallelism to outweigh the thread overhead, so it's not
        # recommended.
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

##
## Buyer beware
##

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
