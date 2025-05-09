FIXME=""" # Update sys.mnoise_S to sys.W
@testitem "Pendulums" begin
    using Markdown
    sys = DPendulum()
    DEFAULTS_SYS = MARX.Pendulums.DEFAULTS_SYS

    # Test the system initialization
    @testset "Initialization" begin
        @test sys.k == 1
        @test sys.Δt == DEFAULTS_SYS.DPendulum.Δt
        @test sys.N == DEFAULTS_SYS.DPendulum.N
        @test sys.z == DEFAULTS_SYS.DPendulum.z
        @test sys.mnoise_S == DEFAULTS_SYS.DPendulum.mnoise_S
        @test sys.mass == DEFAULTS_SYS.DPendulum.mass
        @test sys.length == DEFAULTS_SYS.DPendulum.length
        @test sys.damping == DEFAULTS_SYS.DPendulum.damping
    end

    # Test system query functions
    @testset "System Query Functions" begin
        @test get_state_dim(DPendulum) == DEFAULTS_SYS.DPendulum.D_z
        @test get_observation_dim(DPendulum) == DEFAULTS_SYS.DPendulum.D_y
        @test get_action_dim(DPendulum) == DEFAULTS_SYS.DPendulum.D_a
        @test get_control_dim(DPendulum) == DEFAULTS_SYS.DPendulum.D_u
    end

    # Test inverse inertia matrix calculation
    @testset "Inverse Inertia Matrix" begin
        inv_inertia = MARX.Pendulums.inv_inertia_matrix(sys)
        @test typeof(inv_inertia) == Matrix{Float64}
        @test size(inv_inertia) == (2, 2)
        @test all(isfinite, inv_inertia)  # Ensure finite values in the matrix
    end

    # Test ddzdt dynamics
    @testset "Dynamics Calculation (ddzdt)" begin
        z1, z2 = 0.1, 0.2
        dz1, dz2 = 0.05, -0.05
        u = [0.1, -0.1]

        ddz1, ddz2 = ddzdt(sys, z1, z2, dz1, dz2, u)

        @test typeof(ddz1) == Float64
        @test typeof(ddz2) == Float64
        @test all(isfinite, [ddz1, ddz2])
    end

    # Test state transition
    @testset "State Transition" begin
        z_initial = sys.z
        u = [0.1, -0.1]
        new_state = state_transition!(sys, u)

        @test length(new_state) == get_state_dim(DPendulum)
        @test new_state != z_initial  # Ensure state updates
    end

    # Test measurement function
    @testset "Measurement Function" begin
        y = measure(sys)

        @test length(y) == get_observation_dim(DPendulum)
        @test typeof(y) == Vector{Float64}
        @test all(isfinite, y)
    end

    # Test get_eom_md
    @testset "Get EOM Markdown" begin
        eom_md = get_eom_md(sys)
        @test occursin("System dynamics", eom_md)  # Ensure markdown contains expected content
    end
end
"""
