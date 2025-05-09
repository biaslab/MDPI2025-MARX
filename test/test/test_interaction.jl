@testitem "interaction_MARX" begin
    using MARX

    N = 2
    P = 128
    Δt = 1.0 # 0.01
    ulims = (-500.0, 500.0)
    N_y = 2
    N_u = 3
    T = N*P
    νadd = 2

    for f_env ∈ [MARXSystem]
        D_y = get_observation_dim(f_env)
        D_u = get_control_dim(f_env)
        us_t_beg, us_t_end = 1, D_y * T
        agent = MARXAgent(1, D_y, D_u, ulims; N_y = N_y, N_u = D_u, νadd=νadd)
        env_train, rec_train = interact(f_env, agent, ulims, N, P, Δt, us_t_beg, us_t_end, do_trigger_amnesia=false)
    end
end
