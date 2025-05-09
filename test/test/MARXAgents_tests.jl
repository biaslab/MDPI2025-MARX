@testitem "" begin
    DEFAULTS = MARX.MARXAgents.DEFAULTS
    @testset "MARXAgents Module Tests" begin
        D_y = 3
        D_u = 2
        control_lims = (-1.0, 1.0)
        agent_ID = 1

        # Test MARXAgent Initialization
        agent = MARXAgent(agent_ID, D_y, D_u, control_lims)
        @test agent.ID == agent_ID
        @test agent.D_y == D_y
        @test agent.D_u == D_u
        @test agent.ν == D_y + DEFAULTS.νadd
        @test size(agent.M) == (D_y, agent.D_x)
        @test size(agent.Ω) == (D_y, D_y)
        @test size(agent.Λ) == (agent.D_x, agent.D_x)

        # Test HMARXAgent Initialization
        h_agent = HMARXAgent(agent_ID, D_y, D_u, control_lims)
        @test h_agent.ID == agent_ID
        @test h_agent.D_y == D_y
        @test h_agent.D_u == D_u
        @test h_agent.ν == D_y + DEFAULTS.νadd
        @test h_agent.ν_w == D_y + DEFAULTS.νadd
        @test size(h_agent.M) == (D_y, h_agent.D_x)
        @test size(h_agent.M_w) == (D_y, h_agent.D_x)
        @test size(h_agent.Ω_w) == (D_y, D_y)
        @test size(h_agent.Λ_w) == (h_agent.D_x, h_agent.D_x)

        # Test Update Function for MARXAgent
        initial_ν = agent.ν
        update!(agent)
        @test agent.ν == initial_ν + 1

        # Test Get Estimate Functions
        A_estimate = get_estimate_A(agent)
        W_estimate = get_estimate_W(agent)
        @test size(A_estimate) == size(agent.M)
        @test size(W_estimate) == size(agent.Ω)

        # Test Prediction Function
        μ, y_Σ = predict(agent)
        @test size(μ) == (D_y,)
        @test size(y_Σ) == (D_y, D_y)

        FIXME="""
        # Test PDF Params Function
        log_pdf_value = pdf_params(agent.M, agent.Ω, agent.M, Λ=agent.Λ, Ω=agent.Ω, ν=agent.ν, D_x=agent.D_x, D_y=agent.D_y)
        @test typeof(log_pdf_value) == Float64
        """

        FIXME="""
        # Test PDF Predictive Function
        y = rand(D_y)
        x = rand(agent.D_x)
        pdf_value = pdf_predictive(y, x, agent.M, agent.Λ, agent.Ω, agent.ν, D_y)
        @test typeof(pdf_value) == Float64
        """
    end
end
