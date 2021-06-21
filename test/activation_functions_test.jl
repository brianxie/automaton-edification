using AutomatonEdification

@testset "sigmoid" begin
    @test AutomatonEdification.ActivationFunctions.sigmoid(0) == 0.5
    @test AutomatonEdification.ActivationFunctions.sigmoid(1) == 1/(1 + exp(-1))
    @test AutomatonEdification.ActivationFunctions.sigmoid(2) == 1/(1 + exp(-2))
    @test AutomatonEdification.ActivationFunctions.sigmoid(-1) == 1/(1 + exp(1))

    @test AutomatonEdification.ActivationFunctions.sigmoid([0, 0, 0]) == [0.5, 0.5, 0.5]
    @test AutomatonEdification.ActivationFunctions.sigmoid([0, 1, 2, -1]) ==
        [0.5, 1/(1 + exp(-1)), 1/(1 + exp(-2)), 1/(1 + exp(1))]
end

@testset "relu" begin
    @test AutomatonEdification.ActivationFunctions.relu(0) == 0
    @test AutomatonEdification.ActivationFunctions.relu(1) == 1
    @test AutomatonEdification.ActivationFunctions.relu(-10) == 0

    @test AutomatonEdification.ActivationFunctions.relu([0, 1, -10]) ==
        [0, 1, 0]
end

@testset "softmax" begin
    @test AutomatonEdification.ActivationFunctions.softmax([0]) == [1]
    @test AutomatonEdification.ActivationFunctions.softmax([-2]) == [1]
    @test AutomatonEdification.ActivationFunctions.softmax([1000]) == [1]

    @test AutomatonEdification.ActivationFunctions.softmax([0, 0]) == [0.5, 0.5]
    @test AutomatonEdification.ActivationFunctions.softmax([0, 0, 0, 0]) ==
        [0.25, 0.25, 0.25, 0.25]
    
    @test isapprox(
        AutomatonEdification.ActivationFunctions.softmax([0, -1, 3, 3, -4]),
        [exp(0), exp(-1), exp(3), exp(3), exp(-4)] ./
        (exp(0) + exp(-1) + exp(3) + exp(3) + exp(-4)))
end
