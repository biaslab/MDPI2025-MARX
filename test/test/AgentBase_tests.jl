@testitem "AgentBase" begin
    # Define a mock agent for testing purposes
    mutable struct MockAgent <: OnlineDeterministicAgent
        k::Int
        N_y::Int
        N_u::Int
        D_y::Int
        D_u::Int
        D_x::Int
        ybuffer::RingBuffer{Vector{Float64}}
        ubuffer::RingBuffer{Vector{Float64}}
        memory_type::String
        does_learn::Bool
        does_dream::Bool
        us_matlab::Matrix{Float64}
    end

    # Test for default initialization of an agent
    @testset "Agent Initialization" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        @test agent.k == 1
        @test agent.N_y == N_y
        @test agent.N_u == N_u
        @test agent.D_y == D_y
        @test agent.D_u == D_y
        @test agent.D_x == D_x
        @test agent.memory_type == "dim_first"
        @test agent.does_learn == true
        @test agent.does_dream == true
        @test size(agent.us_matlab) == (2, 10)
    end

    # Test the set_learning! function
    @testset "Learning State Change" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        set_learning!(agent, false)
        @test agent.does_learn == false

        set_learning!(agent, true)
        @test agent.does_learn == true
    end

    # Test the set_dreaming! function
    @testset "Dreaming State Change" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        set_dreaming!(agent, false)
        @test agent.does_dream == false

        set_dreaming!(agent, true)
        @test agent.does_dream == true
    end

    # Test observe! and memorize_observation!
    @testset "Observation and Memorization" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        @test length(agent.ybuffer) == 0
        y = [0.1, 0.2]
        observe!(agent, y)
        @test length(agent.ybuffer) == 1
        @test get_last(agent.ybuffer) == y  # The first element is the observed vector
    end

    # Test memorize_control!
    @testset "Control Memorization" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        @test length(agent.ubuffer) == 0
        u = [0.3, -0.2]
        memorize_control!(agent, u)
        @test length(agent.ubuffer) == 1
        @test get_last(agent.ubuffer) == u  # The second element is the memorized control
    end

    # Test memory function
    @testset "Memory Function" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        ybuffer = RingBuffer(N_y+1, Vector{Float64})
        ubuffer = RingBuffer(N_u, Vector{Float64})
        for _ in 1:N_y+1 push!(ybuffer, zeros(D_y)) end
        for _ in 1:N_u push!(ubuffer, zeros(D_u)) end
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, ybuffer, ubuffer, "dim_first", true, true, rand(2, 10))
        mem = memory(agent)
        @test mem == zeros(D_x)

        y = [0.1, 0.2]
        u = [0.3, -0.2]
        observe!(agent, y)
        memorize_control!(agent, u)

        mem = memory(agent)
        @test typeof(mem) == Vector{Float64}
        @test length(mem) == D_x
    end

    # Test trigger_amnesia!
    @testset "Trigger Amnesia" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        trigger_amnesia!(agent)
        @test length(agent.ybuffer) == N_y + 1
        @test length(agent.ubuffer) == N_u
        #@test get_elements(agent.ybuffer) == zeros(D_y, N_y+1)
        #@test get_elements(agent.ubuffer) == zeros(D_u, N_u)
        @test all(x -> x == zeros(2), get_elements(agent.ybuffer))  # All elements should be zeros
        @test all(x -> x == zeros(2), get_elements(agent.ubuffer))  # All elements should be zeros
    end

    # Test decide! function
    @testset "Decision Making" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        u = decide!(agent)
        @test length(u) == 2  # Control vector should have the same dimension as D_y
        @test agent.k == 2  # The state should increment after a decision
    end

    FIXME="""
    # Test react! function (for both learning and dreaming)
    @testset "React and Learning/Dreaming" begin
        N_y, N_u, D_y, D_u = 3, 3, 2, 2
        D_x = N_y * D_y + N_u * D_u
        agent = MockAgent(1, N_y, N_u, D_y, D_u, D_x, RingBuffer(N_y+1, Vector{Float64}), RingBuffer(N_u, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        y = [0.1, 0.2]
        u = react!(agent, y)

        @test length(u) == 2  # Control vector should have the same dimension as D_y
        @test agent.k == 2  # The state should increment after a reaction
    end
    """

    FIXME="""
    # Test not_implemented functions
    @testset "Not Implemented Functions" begin
        agent = MockAgent(1, 10, 5, 2, 2, RingBuffer(10, Vector{Float64}), RingBuffer(5, Vector{Float64}), "dim_first", true, true, rand(2, 10))

        @test_throws MethodError begin
            params(agent)
        end

        @test_throws MethodError begin
            update!(agent)
        end

        @test_throws MethodError begin
            predict(agent)
        end
    end
    """
end
