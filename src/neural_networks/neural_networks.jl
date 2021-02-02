module NeuralNetworks

# TODO: Add exports

using Zygote

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
    # Each activation function corresponds to a single neuron in this layer.
    activation_fns::AbstractVector{<:Function}
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
    # Partial derivatives w.r.t pre-activation sums.
    dL_dS::AbstractVector{<:Number}
    # Partial derivatives w.r.t each weight.
    dL_dW::AbstractMatrix{<:Number}
    # Partial derivatives w.r.t inputs. This gets used by the previous layer in
    # backprop (these inputs are exactly the outputs of the previous layer).
    dL_dI::AbstractVector{<:Number}
end

struct NeuralNetwork
    layers::AbstractVector{<:Layer}
    loss_fn::Function
end

function Base.show(io::IO, nn::NeuralNetwork)
    num_layers = length(nn.layers)
    print("Neural network with $(num_layers) layers.\n")
    print("Input dimensionality: $(in_ndims_sans_bias(nn.layers[1]))\n")
    print("Output dimensionality: $(out_ndims(nn.layers[num_layers]))\n")
    for l in 1:num_layers
        print("Layer $(l): $(length(nn.layers[l].activation_fns)) neurons.\n")
        weights = nn.layers[l].weights
        print("  Weights [$(size(weights, 1))x$(size(weights, 2))]: $(weights)\n")
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
    create_layer(in_ndims_sans_bias, out_ndims, activation_fns)
"""
function create_layer(in_ndims_sans_bias::Integer,
                      out_ndims::Integer,
                      activation_fns::AbstractVector{<:Function})::Layer
    @assert length(activation_fns) == out_ndims
    weights = randn((in_ndims_sans_bias+1, out_ndims))
    return Layer(weights, activation_fns)
end

"""
    validate_layer(layer)

Asserts that the layer parameters have consistent dimensionality.
"""
function validate_layer(layer::Layer,
                        num_inputs::Integer,
                        num_outputs::Integer)
    @assert size(layer.weights) == (num_inputs + 1, num_outputs)
    @assert length(layer.activation_fns) == num_outputs
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
        expected_in_ndims = (l == 1 ? input_ndims : out_ndims(layers[l-1]))
        actual_in_ndims = in_ndims_sans_bias(layers[l])
        @assert actual_in_ndims ==
            expected_in_ndims "Layer $l: " *
            "in_ndims_sans_bias=$actual_in_ndims, " *
            "expected=$expected_in_ndims"
        if (l == num_layers)
            @assert out_ndims(layers[l]) == output_ndims
        end
    end

    return NeuralNetwork([layers...], loss_fn)
end

"""
    create_nn(layer_dims, activation_fn, input_dims, output_dims)

Creates a neural network by building layers with the specified dimensionality,
`activation_fn`, and `loss_fn`.

- `layer_dims::AbstractVector`: a vector of integers, where the `n`-th value
corresponds to the number of neurons in the `n`-th layer of the network.
- `activation_fn`: the activation function used for all nodes in the network.
- `loss_fn`: single loss function at the end of the network.
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

"""
    forward_pass(layer, inputs)

Computes the activations of `layer` on `inputs`, returning the cached results.
"""
function forward_pass(layer::Layer,
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
    predict(nn, inputs)

Computes the end-to-end application of `nn` on `inputs`.
"""
function predict(nn::NeuralNetwork, inputs::AbstractVector{<:Number})
    for layer in nn.layers
        inputs = forward_pass(layer, inputs).activations
    end
    return inputs
end

"""
    compute_loss_gradient(net_output, label, loss_fn)

Computes the gradient of `loss_fn` with respect to `net_output`, evaluated at
`net_output`.

TODO: Make this work for multiclass loss functions that compute scalar loss from
vector outputs, e.g. softmax.
"""
function compute_loss_gradient(net_output::AbstractVector{<:Number},
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
    compute_loss(net_output, label, loss_fn)

Computes the loss between `label` and `net_output` using `loss_fn`.

TODO: Make this work for multiclass loss functions that compute scalar loss from
vector outputs, e.g. softmax.
"""
function compute_loss(net_output::AbstractVector{<:Number},
                      label::AbstractVector{<:Number},
                      loss_fn::Function)::AbstractVector{<:Number}
    # Vector of partially-applied loss functions, where each entry lossmap[i]
    # contains f(x) = loss_fn(label[i], x)
    lossmap = map(y -> (x -> loss_fn(y, x)), label)

    # Returns a vector loss.
    return map.(lossmap, net_output)
end

"""
    compute_weight_gradients(layer, forward_pass, dL_dO)

Does backprop.

Returns a LayerBackwardPass containing the gradients, without mutating the
original layer.
"""
function compute_weight_gradients(layer::Layer,
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

Updates weights according to backprop gradient calculation.
"""
function update_layer_weights!(layer::Layer,
                               backward_pass::LayerBackwardPass,
                               learning_rate::Number)
    @assert size(layer.weights) == size(backward_pass.dL_dW)

    layer.weights .-= (learning_rate .* backward_pass.dL_dW)
end

"""
Data structure containing loss stats for each epoch of training.
"""
struct TrainingStats
    losses::AbstractVector
end

"""
    train(nn, inputs, labels, learning_rate)

TODO: implement batch and minibatch.
"""
function train!(nn::NeuralNetwork,
                data::AbstractVector,
                labels::AbstractVector,
                learning_rate::Number)::TrainingStats
    @assert length(data) == length(labels)

    num_layers = length(nn.layers)

    # Record the results of forward and backward passes for each layer.
    # These are only used for backprop and are overwritten on every epoch.
    forward_passes = Vector{LayerForwardPass}(undef, num_layers)
    backward_passes = Vector{LayerBackwardPass}(undef, num_layers)

    losses = Vector(undef, length(data))

    for (index, input) in enumerate(data)
        label = labels[index]

        # Run forward-pass.
        for l in 1:num_layers
            forward_passes[l] = forward_pass(nn.layers[l], input)
            input = forward_passes[l].activations
        end

        # Compute loss.
        loss = compute_loss(input, label, nn.loss_fn)

        # Backprop.
        loss_grad = compute_loss_gradient(input, label, nn.loss_fn)

        for l in reverse(1:num_layers)
            backward_passes[l] =
                compute_weight_gradients(nn.layers[l],
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
            print("Iteration $index / $(length(data)) : loss=$(loss)\n")
        end
    end

    return TrainingStats(losses)
end

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
    return size(layer.dL_dW, 1) - 1
end

"""
    in_ndims_sans_bias(layer)

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
    return length(layer.activations)
end

end # module
