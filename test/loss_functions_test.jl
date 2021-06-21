using AutomatonEdification

@testset "mse" begin
    @test AutomatonEdification.LossFunctions.mse([0], [0]) == 0
    @test AutomatonEdification.LossFunctions.mse([1], [-1]) == 4

    @test AutomatonEdification.LossFunctions.mse([-1, -1, -1], [1, 2, 3]) ==
        ((1 - (-1))^2 + (2 - (-1))^2 + (3 - (-1))^2) / 3
end

@testset "cross_entropy" begin
    # TODO: add tests for cross_entropy
end
