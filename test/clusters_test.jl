using AutomatonEdification

@testset "compute_centroid" begin
    @test AutomatonEdification.Clusters.compute_centroid(
        reshape([10], (1,1))) == [10]
    @test AutomatonEdification.Clusters.compute_centroid(
        [10 20]) == [10, 20]
    @test AutomatonEdification.Clusters.compute_centroid([
            10 20 30;
            20 30 40;
            30 40 50;
            60 70 80]) == [30, 40, 50]
end
