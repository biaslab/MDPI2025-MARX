@testitem "Initialization" begin
    #import MARX: Recorder
    @testset "Dimensions" begin
        D_z, D_y, D_a, D_u, D_x, N, K = 3, 2, 4, 2, 5, 10, 3
        recorder = Recorder(D_z, D_y, D_a, D_u, D_x, N, K)

        @test size(recorder.zs) == (D_z, N)
        @test size(recorder.ys) == (D_y, N)
        @test size(recorder.as) == (D_a, N)
        @test size(recorder.us) == (D_u, N)
        @test size(recorder.xs) == (D_x, N)
        @test size(recorder.pred_ys) == (D_y, N, K)
        @test size(recorder.pred_Σs) == (D_y, D_y, N, K)
        @test size(recorder.νs) == (N,)
        @test size(recorder.Λs) == (D_x, D_x, N)
        @test size(recorder.Ωs) == (D_y, D_y, N)
        @test size(recorder.Ws) == (D_y, D_y, N)
        @test size(recorder.Ms) == (D_y, D_x, N)
    end
end

@testitem "Functions" begin
    #import MARX: Recorder
    @testset "record_fn" begin
        D_z, D_y, D_a, D_u, D_x, N, K = 3, 2, 4, 2, 5, 10, 3
        recorder = Recorder(D_z, D_y, D_a, D_u, D_x, N, K)

        # Test recording states
        z_sample = rand(D_z)
        t = 1
        record_zs!(recorder, z_sample, t)
        @test recorder.zs[:, t] == z_sample

        # Test recording observations
        y_sample = rand(D_y)
        record_ys!(recorder, y_sample, t)
        @test recorder.ys[:, t] == y_sample

        # Test recording actions
        a_sample = rand(D_a)
        record_as!(recorder, a_sample, t)
        @test recorder.as[:, t] == a_sample

        # Test recording other fields (similar to above, test one for each dimension type)
        Λ_sample = rand(D_x, D_x)
        record_Λs!(recorder, Λ_sample, t)
        @test recorder.Λs[:, :, t] == Λ_sample
    end

    @testset "predictions" begin
        D_z, D_y, D_a, D_u, D_x, N, K = 3, 2, 4, 2, 5, 10, 3
        recorder = Recorder(D_z, D_y, D_a, D_u, D_x, N, K)

        t, k = 1, 1
        pred_y_sample = rand(D_y)
        pred_Σ_sample = rand(D_y, D_y)

        record_prediction_mean!(recorder, pred_y_sample, t, k)
        @test recorder.pred_ys[:, t, k] == pred_y_sample

        record_prediction_covariance!(recorder, pred_Σ_sample, t, k)
        @test recorder.pred_Σs[:, :, t, k] == pred_Σ_sample
    end
end
