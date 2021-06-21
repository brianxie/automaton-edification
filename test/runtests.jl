using Test

@testset "LossFunctions" begin
    include("clusters_test.jl")
    include("loss_functions_test.jl")
    include("activation_functions_test.jl")
end
