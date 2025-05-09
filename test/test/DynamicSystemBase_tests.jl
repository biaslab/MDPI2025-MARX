@testitem "DynamicSystemBase" begin
    using LinearAlgebra # I
    # Create a mock implementation of DynamicSystem for testing purposes
    mutable struct TestDynamicSystem <: DynamicSystem
        z::Vector{Float64}
        Δt::Float64
        W::Matrix{Float64}
    end

    # Define required functions for the mock system
    function MARX.DynamicSystemBase.get_state_dim(::Type{TestDynamicSystem})
        return 4
    end

    function MARX.DynamicSystemBase.get_observation_dim(::Type{TestDynamicSystem})
        return 2
    end

    function MARX.DynamicSystemBase.inv_inertia_matrix(sys::TestDynamicSystem)
        return I(2)  # Simple identity matrix as inertia
    end

    function MARX.DynamicSystemBase.update_params(sys::TestDynamicSystem) return end

    function MARX.DynamicSystemBase.ddzdt(sys::TestDynamicSystem, z::Vector{Float64}, dz::Vector{Float64}, u::Vector{Float64})
        return -MARX.DynamicSystemBase.inv_inertia_matrix(sys) * u
    end


    # Test for state_transition!
    @testset "State Transition Function" begin
        sys = TestDynamicSystem([1.0, 2.0, 0.1, 0.2], 0.1, I(2))
        u = [0.5, -0.5]

        new_state = state_transition!(sys, u)

        @test length(new_state) == 4
        @test new_state[1:2] ≈ [1.01, 2.02]  # z_new
        @test new_state[3:4] ≈ [0.05, 0.25]  # dz_new
    end

    # Test for measure function
    @testset "Measure Function" begin
        sys = TestDynamicSystem([1.0, 2.0, 0.1, 0.2], 0.1, I(2))

        y = measure(sys)

        @test length(y) == 2
        @test typeof(y) == Vector{Float64}
        @test all(isfinite, y)
    end

    TODO=""" # proper mocking
    # Test for unimplemented functions
    @testset "Unimplemented Abstract Functions" begin
        sys = TestDynamicSystem([1.0, 2.0, 0.1, 0.2], 0.1, I(2))

        @test_throws ErrorException get_eom_md(sys)
        @test_throws ErrorException ddzdt(sys, [1.0, 2.0], [0.1, 0.2], [0.5, -0.5])
        @test_throws ErrorException inv_inertia_matrix(sys)
    end
    """

    TODO=""" # raise ArgumentError instead of ErrorException
    # Test for error checking in state_transition!
    @testset "Error Checking in state_transition!" begin
        sys = TestDynamicSystem([1.0, Inf, 0.1, 0.2], 0.1, I(2))
        u = [0.5, -0.5]

        @test_throws ArgumentError state_transition!(sys, u)
    end
    """
end
