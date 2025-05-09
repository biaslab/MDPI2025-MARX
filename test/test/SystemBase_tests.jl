@testitem "All" begin
    mutable struct MockSystem <: MARX.System
        torque::Vector{Float64}
        z::Vector{Float64}
        Δt::Float64
        N::Int
        k::Int
    end

    # Constructor for MockSystem
    function MockSystem(N::Int, Δt::Float64)
        return MockSystem(zeros(N), zeros(N), Δt, N, 1)
    end

    # Implement required functions for MockSystem
    function get_state_dim(sys::MockSystem)
        return length(sys.z)
    end

    function get_observation_dim(sys::MockSystem)
        return length(sys.z)
    end

    function get_action_dim(sys::MockSystem)
        return length(sys.torque)
    end

    function get_control_dim(sys::MockSystem)
        return length(sys.torque)
    end

    function get_eom_md(sys::MockSystem)
        return "Equation of Motion Model"
    end

    function state_transition!(sys::MockSystem, u::Vector{Float64})
        sys.z .= u  # Just for test purposes, set state to control
    end

    function measure(sys::MockSystem)
        return sum(sys.z)  # Simple measurement, just sum of the state
    end

    # Test for System initialization and basic functionality
    @testset "System Initialization and Basic Functionality" begin
        N, Δt = 10, 0.1
        sys = MockSystem(N, Δt)

        @test typeof(sys) == MockSystem
        @test sys.N == N
        @test sys.Δt == Δt
        @test sys.k == 1
        @test length(sys.torque) == N
        @test length(sys.z) == N
    end

    TODO_FIX="""
    # Test for step! function
    @testset "step! function" begin
        N, Δt = 10, 0.1
        sys = MockSystem(N, Δt)
        a = rand(N)
        ulims = (-1.0, 1.0)

        step!(sys, a, ulims)

        @test sys.k == 2  # Ensure k increments
        @test sys.z == a  # Ensure state transition matches the action (simple case)
    end
    """

    # Test for convert_action function
    @testset "convert_action function" begin
        a = [0.5, -2.0, 1.5, 3.0]
        ulims = (-1.0, 1.0)

        result = convert_action(a, ulims)

        @test all(result .>= -1.0) && all(result .<= 1.0)  # Ensure actions are clamped within the limits
    end

    # Test for get_tsteps function
    @testset "get_tsteps function" begin
        N, Δt = 10, 0.1
        sys = MockSystem(N, Δt)

        tsteps = get_tsteps(sys)

        @test length(tsteps) == N  # Ensure the correct number of time steps
        @test tsteps[1] == 0  # Ensure the first time step is 0
        @test tsteps[end] == (N-1) * Δt  # Ensure the last time step is correct
    end

    # Test for get_control and get_state functions
    @testset "get_control and get_state functions" begin
        N, Δt = 10, 0.1
        sys = MockSystem(N, Δt)

        sys.torque .= [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        sys.z .= [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

        @test get_control(sys) == sys.torque
        @test get_state(sys) == sys.z
    end

    # Test for not implemented error functions
    # TODO: FIX
    """@testset "Not Implemented Error Functions" begin
        sys = MockSystem(10, 0.1)

        @test_throws ErrorException get_state_dim(typeof(sys))  # should throw not implemented error
        @test_throws ErrorException get_observation_dim(typeof(sys))  # should throw not implemented error
        @test_throws ErrorException get_action_dim(typeof(sys))  # should throw not implemented error
        @test_throws ErrorException get_control_dim(typeof(sys))  # should throw not implemented error
        @test_throws ErrorException get_eom_md(sys)  # should throw not implemented error
        @test_throws ErrorException state_transition!(sys, sys.torque)  # should throw not implemented error
        @test_throws ErrorException measure(sys)  # should throw not implemented error
    end"""

end
