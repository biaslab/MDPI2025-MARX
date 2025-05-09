@testitem "LeastSquaresAgents" begin
    import LinearAlgebra: dot
    @testset "OfflineLeastSquaresAgent Tests" begin
        # Initialization test
        agent = OfflineLeastSquaresAgent(1, 2, 3, (-1.0, 1.0))

        @test agent.ID == 1
        @test agent.D_y == 2
        @test agent.D_u == 3
        @test agent.control_lims == (-1.0, 1.0)
        @test size(agent.M) == (2, agent.D_x)

        # params function
        @test params(agent) == agent.M

        # update! test
        D_y, D_x, N = agent.D_y, agent.D_x, 10
        D_z, D_a, D_u, k = agent.D_y, agent.D_u, agent.D_u, 1
        rec = Recorder(D_z, D_y, D_a, D_u, D_x, N, k)
        record_ys!(rec, 5.0*ones(D_y), 1)
        for t in 1:N record_xs!(rec, rand(D_x), t) end
        update!(agent, rec)

        expected_M = (rec.ys * rec.xs') / (rec.xs * rec.xs')
        @test agent.M ≈ expected_M

        # predict test
        predictions = predict(agent, rec)
        @test size(predictions) == (D_y, N)
        @test predictions ≈ agent.M * rec.xs
    end

    @testset "OnlineLeastSquaresAgent Tests" begin
        D_y = 2
        D_u = 3
        # Initialization test
        agent = OnlineLeastSquaresAgent(1, D_y, D_u, (-1.0, 1.0))

        @test agent.ID == 1
        @test agent.D_y == D_y
        @test agent.D_u == D_u
        @test agent.control_lims == (-1.0, 1.0)
        @test size(agent.M) == (agent.D_y, agent.D_x)
        @test length(agent.P) == agent.D_y
        @test all(size(p) == (agent.D_x, agent.D_x) for p in agent.P)

        # params function
        M, P, λ, δ = params(agent)
        @test M == agent.M
        @test P == agent.P
        @test λ == agent.λ
        @test δ == agent.δ

        # update! test
        y = rand(agent.D_y)
        observe!(agent, y)
        @test get_last(agent.ybuffer) == y
        x = memory(agent)

        M_before = copy(agent.M)
        update!(agent)

        for i in 1:agent.D_y
            K_i = agent.P[i] * x / (agent.λ + x' * agent.P[i] * x)
            expected_update = K_i * (y[i] - dot(M_before[i, :], x))
            @test agent.M[i, :] ≈ (M_before[i, :] + expected_update)
        end

        # predict test
        predictions = predict(agent)
        @test size(predictions) == (agent.D_y,)
        @test predictions ≈ agent.M * memory(agent)
    end
end
