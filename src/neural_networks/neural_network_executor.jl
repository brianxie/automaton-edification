include("neural_networks.jl")

# TODO: make this work for batches
using LinearAlgebra, Random

learning_rate = 0.1
abs_error(x, y) = abs(x - y)
e_approx = float(MathConstants.e)
sigmoid(x) = 1.0 / (1.0 + e_approx^(-x))
relu(x) = max(0,x)

# The mapping to learn:
# [0,0] -> [1,0,0,0]
# [0,1] -> [0,2,0,0]
# [1,0] -> [0,0,3,0]
# [1,1] -> [0,0,0,4]

nn = NeuralNetworks.create_nn([2,2,4], relu, abs_error, 2, 4)

for (index, input) in enumerate(shuffle(repeat([[0,0],[0,1],[1,0],[1,1]], 10)))
    print("====\nIteration $index\n====\n")

    # Set up inputs
    val = 1
    if input[1] == 1
        val += 2
    end
    if input[2] == 1
        val += 1
    end
    label = zeros(4)
    label[val] = 1

    print("$input, $label\n")

    # Run inference.
    layer_1_forward = NeuralNetworks.predict(nn.layers[1], input)
    layer_2_forward = NeuralNetworks.predict(nn.layers[2], layer_1_forward.activations)
    layer_3_forward = NeuralNetworks.predict(nn.layers[3], layer_2_forward.activations)

    print("Prediction: $(layer_3_forward.activations)\n")

    loss_grad = NeuralNetworks.compute_loss_gradient(layer_3_forward.activations,
                                                     label,
                                                     abs_error)
    layer_3_backward =
        NeuralNetworks.compute_weight_gradients(nn.layers[3],
                                                layer_3_forward,
                                                loss_grad)
    layer_2_backward =
        NeuralNetworks.compute_weight_gradients(nn.layers[2],
                                                layer_2_forward,
                                                layer_3_backward.dL_dI[1:end-1])
    layer_1_backward =
        NeuralNetworks.compute_weight_gradients(nn.layers[1],
                                                layer_1_forward,
                                                layer_2_backward.dL_dI[1:end-1])

    print("Layer 3 ")
    nn.layers[3] = NeuralNetworks.update_layer_weights(nn.layers[3],
                                                       layer_3_backward,
                                                       learning_rate)
    print("Layer 2 ")
    nn.layers[2] = NeuralNetworks.update_layer_weights(nn.layers[2],
                                                       layer_2_backward,
                                                       learning_rate)
    print("Layer 1 ")
    nn.layers[1] = NeuralNetworks.update_layer_weights(nn.layers[1],
                                                       layer_1_backward,
                                                       learning_rate)

end
