@testitem "MARXSystem" begin
    using Markdown
    DEFAULTS_SYS = MARX.MARXSystems.DEFAULTS_SYS
    @testset "MARXSystem Initialization" begin
        sys = MARXSystem()

        D_y = DEFAULTS_SYS.MARXSystem.D_y

        @test sys.k == 1
        @test sys.Δt == DEFAULTS_SYS.MARXSystem.Δt
        @test sys.N == DEFAULTS_SYS.MARXSystem.N
        @test sys.z == DEFAULTS_SYS.MARXSystem.z
        @test sys.W == DEFAULTS_SYS.MARXSystem.W
        @test sys.memory_type == DEFAULTS_SYS.MARXSystem.memory_type
        @test sys.D_x > 0
        @test size(sys.A) == (D_y, sys.D_x)
    end

    # Test for System Query Functions
    @testset "System Query Functions" begin
        @test get_state_dim(MARXSystem) == DEFAULTS_SYS.MARXSystem.D_z
        @test get_observation_dim(MARXSystem) == DEFAULTS_SYS.MARXSystem.D_y
        @test get_action_dim(MARXSystem) == DEFAULTS_SYS.MARXSystem.D_a
        @test get_control_dim(MARXSystem) == DEFAULTS_SYS.MARXSystem.D_u
        @test typeof(get_eom_md(MARXSystem())) == Markdown.MD
    end

    TODO="""
    # Test for memorize_observation! and memorize_control!
    @testset "Memory Functions" begin
        sys = MARXSystem()
        obs = [1.0, 2.0]
        ctrl = [0.5, -0.5]

        memorize_observation!(sys, obs)
        memorize_control!(sys, ctrl)

        @test get_last(sys.ybuffer) == obs
        @test get_last(sys.ubuffer) == ctrl
    end

    # Test for memory_system
    @testset "Memory System Function" begin
        sys = MARXSystem()
        obs = [1.0, 2.0]
        ctrl = [0.5, -0.5]

        memorize_observation!(sys, obs)
        memorize_control!(sys, ctrl)

        mem = memory_system(sys)
        D_x = DEFAULTS_SYS.MARXSystem.D_y * sys.N_y + DEFAULTS_SYS.MARXSystem.D_u * sys.N_u

        @test length(mem) == D_x
        @test mem[end] == ctrl[end]
    end
    """

    # Test for state_transition!
    @testset "State Transition" begin
        sys = MARXSystem()
        ctrl = [0.5, -0.5]

        new_state = state_transition!(sys, ctrl)

        @test length(new_state) == DEFAULTS_SYS.MARXSystem.D_z
        @test sys.z == new_state
    end

    # Test for measure
    @testset "Measurement Function" begin
        sys = MARXSystem()

        y = measure(sys)
        D_y = DEFAULTS_SYS.MARXSystem.D_y

        @test length(y) == D_y
    end
end
