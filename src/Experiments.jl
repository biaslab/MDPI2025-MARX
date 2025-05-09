module Experiments

using ..AgentBase
using ..MARXAgents
using ..LeastSquaresAgents
using ..SystemBase
import ..SystemBase: get_observation_dim, get_control_dim, measure, step!
using ..MARXSystems
using ..Pendulums
using ..MassSpringSystems
using ..InteractionRecorder

using Distributions # mean
using JLD2 # saving stuff
#using Threads # @threads

export interact, train_test_run
export monte_carlo_experiment, monte_carlo_experiment_T, monte_carlo_experiment_model_selection_Nys_Nus

function predict_and_record!(rec::Recorder, agent::Agent, t::Int, k::Int)
    # TODO: END react code
    if agent isa OnlineDeterministicAgent
        # Call predict for a deterministic agent (expect only the mean)
        pred_y = predict(agent)
    elseif agent isa OnlineProbabilisticAgent
        # Call predict for a probabilistic agent (expect mean and covariance)
        pred_y, pred_Σ = predict(agent)
        record_prediction_covariance!(rec, pred_Σ, t, k)
    else
        error("Unknown agent type")
    end
    record_prediction_mean!(rec, pred_y, t, k)
    return pred_y
end

# TODO: use react! function of agent
function react_and_record!(rec::Recorder, agent::Agent, y::Vector{Float64}, pred_y::Vector{Float64}, t::Int)
    #a = react!(agent, y)

    # TODO: update before or after control decision?? should be before but in the example notebook its after?
    #println("measured: $y $(agent.does_dream)")
    observe!(agent, agent.does_dream ? pred_y : y)
    record_xs!(rec, memory(agent), t)
    pred_y = predict_and_record!(rec, agent, t, 1)
    if agent isa AbstractMARXAgent
        record_surprisals!(rec, surprisal(agent), t)
        record_vfes!(rec, variational_free_energy(agent), t)
        record_ces_posterior_prior!(rec, ce_posterior_prior(agent), t)
        record_ces_posterior_likelihood!(rec, ce_posterior_likelihood(agent), t)
        record_es_posterior!(rec, e_posterior(agent), t)
        record_kls_posterior_prior!(rec, kl_posterior_prior(agent), t)
        record_kls_posterior_likelihood!(rec, kl_posterior_likelihood(agent), t)
    end
    agent.does_learn && update!(agent)

    if agent isa AbstractMARXAgent
        ν, Λ, Ω, M = AgentBase.params(agent)
        record_νs!(rec, ν, t)
        record_Λs!(rec, Λ, t)
        record_Ωs!(rec, Ω, t)
        record_Ws!(rec, get_estimate_W(agent), t)
    elseif typeof(agent) == OnlineLeastSquaresAgent
        M, P, λ, δ = AgentBase.params(agent)
    else
        M = AgentBase.params(agent)
    end
    record_Ms!(rec, M, t)
    a = decide!(agent)
    record_as!(rec, a, t+1)
    return a, pred_y
end

# TODO: there was some fill_memory code which I removed. Maybe filling the memory makes sense at some point
function interact(f_env::DataType, agent::Agent, ulims::Tuple{Float64, Float64}, N_fmultisingen::Int, P_fmultisingen::Int, Δt::Float64, t_beg::Int, t_end::Int; is_learning::Bool=true, is_dreaming::Bool=false, do_trigger_amnesia::Bool=true, a_scale::Float64=1.0)
    N = N_fmultisingen*P_fmultisingen
    env = f_env(N = N, Δt=Δt)
    return interact(env, agent, ulims, N_fmultisingen, P_fmultisingen, Δt, t_beg, t_end, is_learning=is_learning, is_dreaming=is_dreaming, do_trigger_amnesia=do_trigger_amnesia, a_scale=a_scale)
end

function interact(env::System, agent::Agent, ulims::Tuple{Float64, Float64}, N_fmultisingen::Int, P_fmultisingen::Int, Δt::Float64, t_beg::Int, t_end::Int; is_learning::Bool=true, is_dreaming::Bool=false, do_trigger_amnesia::Bool=true, a_scale::Float64=1.0)
    N = N_fmultisingen*P_fmultisingen
    f_env = typeof(env)
    D_z, D_y, D_a, D_u, D_x = get_state_dim(f_env), get_observation_dim(f_env), get_action_dim(f_env), get_control_dim(f_env), agent.D_x
    rec = Recorder(D_z, D_y, D_a, D_u, D_x, N, 1)

    do_trigger_amnesia && trigger_amnesia!(agent)
    #if !is_learning trigger_amnesia!(agent) end
    set_learning!(agent, is_learning)
    override_controls!(agent, N_fmultisingen, P_fmultisingen, t_beg, t_end, a_scale=a_scale)
    set_dreaming!(agent, is_dreaming)

    # TODO: record z_0

    a = decide!(agent) # a_{t-1} from agent's decision-making process
    record_as!(rec, a, 1)
    u = convert_action(a, ulims)
    record_us!(rec, u, 1)
    #println(typeof(rec.As))
    #println(size(rec.As))
    #println(size(get_state_transition_matrix(env)))
    env isa AbstractMARXSystem && record_As!(rec, get_state_transition_matrix(env), 1)
    step!(env, u, ulims) # z_t = f(z_{t-1}, u_{t-1})
    pred_y = zeros(D_y)
    for t = 1:N-1
        # system is now in z_t
        record_zs!(rec, get_state(env), t)
        y = measure(env) # y_t = g(z_t)
        record_ys!(rec, y, t)
        a, pred_y = react_and_record!(rec, agent, y, pred_y, t) # u_t = clip(a_t) from agents decision-making process
        u = convert_action(a, ulims)
        record_us!(rec, u, t+1)
        env isa AbstractMARXSystem && record_As!(rec, get_state_transition_matrix(env), t+1)
        step!(env, u, ulims) # z_t = f(z_{t-1}, u_{t-1})
    end
    record_zs!(rec, get_state(env), N)
    y = measure(env)
    record_ys!(rec, y, N)
    if agent isa AbstractMARXAgent
        record_surprisals!(rec, surprisal(agent), N)
        record_vfes!(rec, variational_free_energy(agent), N)
        record_ces_posterior_prior!(rec, ce_posterior_prior(agent), N)
        record_ces_posterior_likelihood!(rec, ce_posterior_likelihood(agent), N)
        record_es_posterior!(rec, e_posterior(agent), N)
        record_kls_posterior_prior!(rec, kl_posterior_prior(agent), N)
        record_kls_posterior_likelihood!(rec, kl_posterior_likelihood(agent), N)
    end
    observe!(agent, agent.does_dream ? pred_y : y)
    record_xs!(rec, memory(agent), N)
    agent.does_learn && update!(agent)
    k = 1
    # TODO: END react code
    if agent isa OnlineDeterministicAgent
        # Call predict for a deterministic agent (expect only the mean)
        pred_y = predict(agent)
    elseif agent isa OnlineProbabilisticAgent
        # Call predict for a probabilistic agent (expect mean and covariance)
        pred_y, pred_Σ = predict(agent)
        record_prediction_covariance!(rec, pred_Σ, N, k)
    else
        error("Unknown agent type")
    end
    record_prediction_mean!(rec, pred_y, N, k)
    if agent isa AbstractMARXAgent
        ν, Λ, Ω, M = AgentBase.params(agent)
        record_νs!(rec, ν, N)
        record_Λs!(rec, Λ, N)
        record_Ωs!(rec, Ω, N)
        record_Ws!(rec, get_estimate_W(agent), N)
    elseif typeof(agent) == OnlineLeastSquaresAgent
        M, P, λ, δ = AgentBase.params(agent)
    else
        M = AgentBase.params(agent)
    end
    record_Ms!(rec, M, N)
    #a = react_and_record!(rec, agent, y, N)
    return env, rec
end

function train_offline()
    # TODO
    agent = MARXAgent(1, D_y, D_u, ulims; N_y = hyperparams.N_y, N_u = hyperparams.N_u, νadd=hyperparams.νadd)
    # run one agent s.t. we can get the observations for offline training (it does not matter which agent runs
    env_train, rec_train = interact(f_env, agent, ulims, N, P, Δt, t_beg_train, t_end_train, do_trigger_amnesia=false)
end

function train_test_run(f_env::DataType, f_agent::DataType, N::Int, P::Int, N_test::Int, P_test::Int, Δt::Float64, ulims::Tuple{Float64, Float64}, N_y::Int, N_u::Int, νadd::Int, t_beg_train::Int, t_end_train::Int, t_beg_test::Int, t_end_test::Int, agent_idx::Int, cΩ::Float64, cΛ::Float64)
    T_train = N*P
    T_test = N_test*P_test

    D_y, D_u = get_observation_dim(f_env), get_action_dim(f_env)
    if f_agent == MARXAgent
        agent = f_agent(agent_idx, D_y, D_u, ulims; N_y = N_y, N_u = N_u, νadd=νadd, cΩ=cΩ, cΛ=cΛ)
    elseif f_agent == OnlineLeastSquaresAgent
        agent = f_agent(agent_idx, D_y, D_u, ulims; N_y = N_y, N_u = N_u)
    else
        throw(ArgumentError("Unsupported agent type: $(f_agent)"))
    end
    # train the agent
    env_train, rec_train = interact(f_env, agent, ulims, N, P, Δt, t_beg_train, t_end_train, do_trigger_amnesia=false)
    # test the agent
    env_test, rec_test = interact(f_env, agent, ulims, N_test, P_test, Δt, t_beg_test, t_end_test; is_learning=false, do_trigger_amnesia=true)
    return agent, env_train, rec_train, env_test, rec_test
end

function monte_carlo_experiment(N_runs::Int, f_env::DataType, f_agent::DataType, N::Int, P::Int, Δt::Float64, ulims::Tuple{Float64, Float64}, N_y::Int, N_u::Int, νadd::Int, cΩ::Float64, cΛ::Float64)
    us_t_total = 327680
    T = N*P
    N_test = 50
    P_test = 2
    T_test = N_test*P_test
    D_y = get_observation_dim(f_env)
    agents = Vector{Agent}(undef, N_runs)
    envs_train, envs_test = Vector{f_env}(undef, N_runs), Vector{f_env}(undef, N_runs)
    recs_train, recs_test = Vector{Recorder}(undef, N_runs), Vector{Recorder}(undef, N_runs)
    abs_errors = zeros(D_y, T_test, N_runs)
    rmses = zeros(D_y, N_runs)
    for i in 1:N_runs
        us_train_offset_train = (i-1)*T
        us_train_offset_test = (i-1)*T_test
        us_t_beg_train, us_t_end_train = 1 + us_train_offset_train, D_y*T + us_train_offset_train
        us_t_beg_test, us_t_end_test = us_t_total - D_y*T_test+1 - D_y*T_test*N_runs + us_train_offset_test, us_t_total - D_y*T_test*N_runs + us_train_offset_test
        agents[i], envs_train[i], recs_train[i], envs_test[i], recs_test[i] = train_test_run(f_env, f_agent, N, P, N_test, P_test, Δt, ulims, N_y, N_u, νadd, us_t_beg_train, us_t_end_train, us_t_beg_test, us_t_end_test, i, cΩ, cΛ)
        rec_test = recs_test[i]
        abs_errors[:,:,i] = rec_test.ys .- rec_test.pred_ys[:,:,1]
        rmses[:, i] = [ sqrt(mean(abs_errors[dim,:,i].^2)) for dim in 1:D_y ]
    end
    return agents, envs_train, envs_test, recs_train, recs_test, rmses
end

function monte_carlo_experiment_T(label_env::String, label_agent::String, Ts::Vector{Int}, N_runs::Int, f_env::DataType, f_agent::DataType, Δt::Float64, ulims::Tuple{Float64, Float64}, N_y::Int, N_u::Int, νadd::Int, cΩ::Float64, cΛ::Float64)
    n_Ts = length(Ts)
    for (i, T) in enumerate(Ts)
        println("$label_env $label_agent $T")
        P = 2
        N = Int(T/P)
        agent, env_train, env_test, rec_train, rec_test, rmse = monte_carlo_experiment(N_runs, f_env, f_agent, N, P, Δt, ulims, N_y, N_u, νadd, cΩ, cΛ)
        f_name = "results/$(label_env)/$(label_agent)_monte_carlo_T$(T).jld2"
        @save f_name agent=agent env_train=env_train env_test=env_test rec_train=rec_train rec_test=rec_test rmse=rmse
    end
end

function monte_carlo_experiment_T_old(Ts::Vector{Int}, N_runs::Int, f_env::DataType, f_agent::DataType, Δt::Float64, ulims::Tuple{Float64, Float64}, N_y::Int, N_u::Int, νadd::Int, cΩ::Float64, cΛ::Float64)
    n_Ts = length(Ts)
    D_y = get_observation_dim(f_env)
    rmses = zeros(D_y, N_runs, n_Ts)
    agents = Matrix{Agent}(undef, N_runs, n_Ts)
    envs_train, envs_test = Matrix{f_env}(undef, N_runs, n_Ts), Matrix{f_env}(undef, N_runs, n_Ts)
    recs_train, recs_test = Matrix{Recorder}(undef, N_runs, n_Ts), Matrix{Recorder}(undef, N_runs, n_Ts)
    for (i, T) in enumerate(Ts)
        P = 2
        N = Int(T/P)
        #println("$i $T $N_runs $N $P $N_y $N_u")
        agents[:,i], envs_train[:,i], envs_test[:,i], recs_train[:,i], recs_test[:,i], rmses[:,:,i] = monte_carlo_experiment(N_runs, f_env, f_agent, N, P, Δt, ulims, N_y, N_u, νadd, cΩ, cΛ)
    end
    return agents, envs_train, envs_test, recs_train, recs_test, rmses
end

function monte_carlo_experiment_model_selection_Nys_Nus(Nys::Vector{Int}, Nus::Vector{Int}, N_runs::Int, f_env::DataType, f_agent::DataType, N::Int, P::Int, Δt::Float64, ulims::Tuple{Float64, Float64}, νadd::Int, cΩ::Float64, cΛ::Float64)
    D_y = get_observation_dim(f_env)
    n_Nys = length(Nys)
    n_Nus = length(Nus)
    rmses = zeros(D_y, N_runs, n_Nys, n_Nus)
    agents = Array{Agent}(undef, N_runs, n_Nys, n_Nus)
    envs_train, envs_test = Array{f_env}(undef, N_runs, n_Nys, n_Nus), Array{f_env}(undef, N_runs, n_Nys, n_Nus)
    recs_train, recs_test = Array{Recorder}(undef, N_runs, n_Nys, n_Nus), Array{Recorder}(undef, N_runs, n_Nys, n_Nus)
    for (i, N_y) in enumerate(Nys)
        for (j, N_u) in enumerate(Nus)
            agents[:,i,j], envs_train[:,i,j], envs_test[:,i,j], recs_train[:,i,j], recs_test[:,i,j], rmses[:,:,i,j] = monte_carlo_experiment(N_runs, f_env, f_agent, N, P, Δt, ulims, N_y, N_u, νadd, cΩ, cΛ)
        end
    end
    return agents, envs_train, envs_test, recs_train, recs_test, rmses
end

end # module
