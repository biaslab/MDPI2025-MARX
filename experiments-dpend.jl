using MARX
using BenchmarkTools

f_env = DPendulum
#f_env = DoubleMassSpringDamperSystem
f_agent = MARXAgent
label_env = string(nameof(f_env))
Δt, N_y, N_u, νadd, ulims = 0.05, 5, 5, 2, (-500.0, 500.0)
cΩ_weak, cΛ_weak, cΩ_uninformative, cΛ_uninformative = 1e-2, 1e-1, 1e-5, 1e-4
Ts = collect(2:2:64)
N_runs = 600
T = Ts[end]
P = 2
N = Int(T / P)

@time agent, env_train, env_test, rec_train, rec_test = monte_carlo_experiment(N_runs, f_env, f_agent, N, P, Δt, ulims, N_y, N_u, νadd, cΩ_weak, cΛ_weak)

#@time monte_carlo_experiment_T_parallel_live(label_env, "MARX-WI", Ts, N_runs, f_env, MARXAgent, Δt, ulims, N_y, N_u, νadd, cΩ_weak, cΛ_weak)
#@time monte_carlo_experiment_T_parallel_live(label_env, "MARX-UI", Ts, N_runs, f_env, MARXAgent, Δt, ulims, N_y, N_u, νadd, cΩ_uninformative, cΛ_uninformative)
#@time monte_carlo_experiment_T_parallel_live(label_env, "RLS", Ts, N_runs, f_env, OnlineLeastSquaresAgent, Δt, ulims, N_y, N_u, νadd, cΩ_uninformative, cΛ_uninformative)
