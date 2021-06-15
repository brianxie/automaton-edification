using AutomatonEdification

# Tests for mse.
@testset "mse" begin
    @test AutomatonEdification.LossFunctions.mse([0], [0]) == 0
    @test AutomatonEdification.LossFunctions.mse([-1], [1]) == 4
end
