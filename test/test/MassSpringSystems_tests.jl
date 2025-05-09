@testitem "MassSpringSystems" begin
    import LinearAlgebra: diagm
    DEFAULTS_SYS = MARX.MassSpringSystems.DEFAULTS_SYS
    sys = DoubleMassSpringDamperSystem()

    # Test the system initialization
    @testset "Initialization" begin
        @test sys.k == 1
        @test sys.Δt == DEFAULTS_SYS.DoubleMassSpringDamperSystem.Δt
        @test sys.N == DEFAULTS_SYS.DoubleMassSpringDamperSystem.N
        @test sys.z == DEFAULTS_SYS.DoubleMassSpringDamperSystem.z
        @test sys.W == DEFAULTS_SYS.DoubleMassSpringDamperSystem.W
        @test sys.mass == DEFAULTS_SYS.DoubleMassSpringDamperSystem.mass
        @test sys.spring == DEFAULTS_SYS.DoubleMassSpringDamperSystem.spring
        @test sys.damping == DEFAULTS_SYS.DoubleMassSpringDamperSystem.damping
    end

    # Test system query functions
    @testset "System Query Functions" begin
        @test get_state_dim(DoubleMassSpringDamperSystem) == DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_z
        @test get_observation_dim(DoubleMassSpringDamperSystem) == DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_y
        @test get_action_dim(DoubleMassSpringDamperSystem) == DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_a
        @test get_control_dim(DoubleMassSpringDamperSystem) == DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_u
    end

    # Test inverse inertia matrix calculation
    @testset "Inverse Inertia Matrix" begin
        inv_inertia = MARX.MassSpringSystems.inv_inertia_matrix(sys)
        @test typeof(inv_inertia) == Matrix{Float64}
        @test inv_inertia == diagm(1 ./ sys.mass)
    end

    # Test ddzdt dynamics
    @testset "Dynamics Calculation (ddzdt)" begin
        z = [0.1, 0.2]
        dz = [0.05, -0.05]
        u = [0.1, -0.1]

        ddz = ddzdt(sys, z, dz, u)

        @test length(ddz) == 2
        @test typeof(ddz) == Vector{Float64}
        @test all(isfinite, ddz)
    end

    # Test state transition
    @testset "State Transition" begin
        z_initial = sys.z
        u = [0.1, -0.1]
        new_state = state_transition!(sys, u)

        @test length(new_state) == get_state_dim(DoubleMassSpringDamperSystem)
        @test new_state != z_initial  # Ensure state updates
    end

    # Test measurement function
    @testset "Measurement Function" begin
        y = measure(sys)

        @test length(y) == get_observation_dim(DoubleMassSpringDamperSystem)
        @test typeof(y) == Vector{Float64}
        @test all(isfinite, y)
    end

    # Test get_eom_md
    @testset "Get EOM Markdown" begin
        eom_md = string(get_eom_md(sys))
        @test occursin("System dynamics", eom_md)  # Ensure markdown contains expected content
    end
end
