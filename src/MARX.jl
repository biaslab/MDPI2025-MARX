module MARX

# Miscellanous helper functions
include("RingBuffers.jl")
include("Utils.jl")
# Storage and Memory
include("InteractionRecorder.jl")
# Systems
include("SystemBase.jl")
include("MARXSystems.jl")
## Systems where the equation of motion a set of ODEs solved by numerical procedures (forward euler)
include("DynamicSystemBase.jl")
include("MassSpringSystems.jl")
include("Pendulums.jl") # TODO fix
# Agents
include("AgentBase.jl")
include("LeastSquaresAgents.jl")
include("MARXAgents.jl")
# Interaction and Plotting
include("Experiments.jl")
include("Plotting.jl")

# RingBuffers
export RingBuffer, get_elements, get_elements_reverse, get_vector, get_last, length
# Utils
export ui_choices, rectangle, calculate_cart_positions, check_nan_or_inf, has_nan_or_inf, polar2cart, cart2polar, generateMultisineExcitation, load_generated_controls, getButtcoefs, get_memory_ticks, mvgamma, logmvgamma, mvdigamma, logmvdigamma, not_implemented_error, setup_system_matrix, initialize_buffer, compute_memory_size
# SystemBase
export System, step!, update!, get_realtime_xticks, get_state_dim, get_observation_dim, get_action_dim, get_control_dim, get_eom_md, state_transition!, measure, get_control, get_state, get_tsteps, convert_action
# DynamicSystemBase
export DynamicSystem, ddzdt, update_params
# MARXSystems
export AbstractMARXSystem, MARXSystem, TvMARXSystem
export get_state_transition_matrix, get_process_noise_matrix
# MassSpringSystems
export DoubleMassSpringDamperSystem
# Pendulums
export DPendulum
# AgentBase
export Agent, OnlineDeterministicAgent, OfflineDeterministicAgent, OnlineProbabilisticAgent, OfflineProbabilisticAgent
export set_learning!, set_dreaming!, override_controls!, observe!, memorize_observation!, memory, trigger_amnesia!, decide!, memorize_control!, react!
export params, predict, update!
# LeastSquaresAgent
export OfflineLeastSquaresAgent, OnlineLeastSquaresAgent
# MARXAgents
export AbstractMARXAgent, MARXAgent, HMARXAgent
export get_estimate_A, get_estimate_W, pdf_params, pdf_predictive, pdf_likelihood, surprisal, variational_free_energy
export ce_posterior_prior, e_posterior, ce_posterior_likelihood
export kl_posterior_prior, kl_posterior_likelihood
# InteractionRecorder
export Recorder
export record_zs!, record_ys!, record_as!, record_us!, record_xs!, record_νs!, record_Λs!, record_Ωs!, record_Ws!, record_Ms!
export record_surprisals!, record_vfes!, record_ces_posterior_prior!, record_ces_posterior_likelihood!, record_es_posterior!, record_kls_posterior_prior!, record_kls_posterior_likehood!
export record_prediction_mean!, record_prediction_covariance!
# Experiments
export interact, train_test_run
export monte_carlo_experiment, monte_carlo_experiment_T, monte_carlo_experiment_T_parallel, monte_carlo_experiment_T_parallel_live
export load_experiment_results, prepare_experiment_data
# Plotting
export plot_controls, plot_observations, plot_trial_test_dream
export plot_error
export plot_system, animate_system, plot_predictions, plot_monte_carlo_results, plot_agent_performance, plot_agent_params, plot_params
export plot_param_M_timeseries, plot_param_M_compare_timeseries
export plot_param_M, plot_param_M_combos, plot_update_M
export plot_param_A_norm, plot_param_W_norm, plot_param_A_norm_old, plot_param_AW_norm
export violinplot_param_A_norm, violinplot_param_W_norm, violinplot_param_AW_norm
export plot_pdf_params
export plots_paper, plots_paper_single
export plot_pdf_predictive
export plot_surprisals, plot_ces_posterior_likelihood, plot_ces_posterior_prior, plot_es_posterior
export plot_ces_posterior_likelihood_kls_posterior_prior, plot_ces_posterior_likelihood_ces_posterior_prior_es_posterior
export plot_param_W_timeseries
export plot_param_Ω, plot_param_Ω_combos, plot_update_Ω, plot_param_Ω_timeseries
export plot_param_Λ, plot_param_Λ_combos, plot_update_Λ, plot_param_Λ_timeseries, plot_param_Λ_timeseries_dim
export plot_param_ΛΩ
export plot_system_params_A, plot_params_A_comparison, plot_params_A_diff
export plot_system_params_W, plot_params_W_comparison
export plot_det_ΛΩ
export plot_rmse_boxplot, plot_rmse_line, plot_rmse_line_baseline, plot_rmse_violin
export plot_rmse_heatmap_Nys_Nus

using .RingBuffers
using .Utils
using .SystemBase
using .DynamicSystemBase
using .MARXSystems
using .MassSpringSystems
using .Pendulums
using .AgentBase
using .LeastSquaresAgents
using .MARXAgents
using .InteractionRecorder
using .Experiments
using .Plotting

end # module MARX
