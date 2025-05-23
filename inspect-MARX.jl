### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a3fd9a34-ed1a-4b78-aca0-acf7bfc99a7b
begin
using Pkg
Pkg.update()
Pkg.add(url="https://github.com/biaslab/MDPI2025-MARX#main")
using MARX
using JLD2
end

# ╔═╡ e389e050-65e2-11ef-3cae-03e1bfe027f0
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(80px, 5%);
    	padding-right: max(80px, 5%);
	}
</style>
"""

# ╔═╡ b7debaa5-8ebd-4101-816c-9a44f49c40e9
begin
	dir_figures = "figures/"
	mkpath(dir_figures)
	f_env = MARXSystem
	label_env = string(nameof(f_env))
	D_y = get_observation_dim(f_env)
	Ts = collect(2:2:64)
	n_Ts = length(Ts)
	N_runs = 600
	dpi = 300
	methods = ["MARX-WI", "MARX-UI", "RLS"]
end

# ╔═╡ 6899ed0a-d995-4c26-8a0c-a7a58b9d02e7
md"# Testing: Monte Carlo experiments for different training sizes"

# ╔═╡ 636f758f-f182-4d11-b29c-ae2e46ef47f2
@time results, rmses_all = prepare_experiment_data(f_env, methods, Ts, N_runs, D_y)

# ╔═╡ 99a54091-3da8-4964-9b36-7e4e501579d0
md"### True system parameters (of a single monte carlo run for the MARX-WI method)"

# ╔═╡ 897f2a85-1353-43e6-bdc6-6d1f2f07c1e1
plot_system_params_A(results["MARX-WI"].envs_train[1], results["MARX-WI"].agents[1], psize=(4*dpi,0.6dpi), f_name="figures/$(label_env)-coeffs-A-system.png")

# ╔═╡ a0a31c51-2a1c-4251-adf8-2ed10272ac2c
md"### Simulation errors (average RMSE with standard error) of all three estimators)"

# ╔═╡ b45c953c-36cb-497d-9968-d139392bf441
plot_rmse_line_baseline(rmses_all, Ts, f_name="figures/$(label_env)-montecarlo-ribbonedline-baseline-sum.png", logscale=false)

# ╔═╡ 94836524-fdf7-4bcf-a65a-3343f724a196
md"# DEBUGGING: Inspecting performances for a single run"

# ╔═╡ 7b412a6c-a8a5-4c93-8083-11acbd1e5567
@bind hyperparams_debug MARX.ui_choices([
	Dict(:type => :Slider, :label => "T", :range => 1:length(Ts), :default => length(Ts))
])

# ╔═╡ e98d41f6-0d08-4677-b283-d589b7b9c2a5
begin
	# TODO: use @view to avoid allocations. resolve: "invalid use of @view macro"
	methods_MARX = ["MARX-WI", "MARX-UI"]
	pdf_param_data = [(label, results[label].recs_train[:, hyperparams_debug.T], results[label].envs_train[:, hyperparams_debug.T]) for label in methods_MARX]

    param_norm_A_data = [(label, results[label].recs_train[:, hyperparams_debug.T], results[label].envs_train[:, hyperparams_debug.T]) for label in methods]
	param_norm_W_data = [(label, results[label].recs_train[:, hyperparams_debug.T], results[label].envs_train[:, hyperparams_debug.T]) for label in methods_MARX]
end

# ╔═╡ c99307a2-cfbf-471e-8a3b-0d4c694ea712


# ╔═╡ 9c9729a3-226b-493f-9ee8-4104ff6ff8c5
plot_pdf_params(pdf_param_data, f_name="figures/$(label_env)-pdf-params.png")

# ╔═╡ 2792d246-0377-40f7-adf5-ef99d023aa9f
plot_param_A_norm(param_norm_A_data, logscale=true, f_name="figures/$(label_env)-A-norm.png")

# ╔═╡ 87614be1-7870-444d-b985-8dda3844aa2f
plot_param_W_norm(param_norm_W_data, logscale=true, f_name="figures/$(label_env)-W-norm.png")

# ╔═╡ 408cb4a4-3e56-45e9-9b08-cf43116ceb18
plot_pdf_predictive(pdf_param_data, f_name="figures/$(label_env)-pdf-predictive.png")

# ╔═╡ 124996b5-8df5-4207-9950-ca700c8a0ade
violinplot_param_AW_norm(param_norm_A_data; logscale=true, psize=(1200, 500))

# ╔═╡ d88f5e6b-3afd-456f-a796-2b316cbe7cc3
md"## DEBUGGING: Inspecting performance of MARX-WI of a single run for T=$(hyperparams_debug.T)"

# ╔═╡ 0bb97d07-7628-4973-8bf9-cbfb96438cfa
@bind hyperparams_run_WI MARX.ui_choices([
	Dict(:type => :Slider, :label => "run", :range => 1:N_runs, :default => 1),
])

# ╔═╡ ab00b2e0-b28a-4224-92d4-add711923ff1
plot_param_W_timeseries(results["MARX-WI"].recs_train[hyperparams_run_WI.run,hyperparams_debug.T], sys=results["MARX-WI"].envs_train[hyperparams_run_WI.run,hyperparams_debug.T], f_name="figures/$(label_env)-coeffs-W-timeseries")

# ╔═╡ 425cb22a-def8-4aa5-bb27-96b2ad0b9e09
A_indices = [(1,1), (2,1), (1,6), (2,6)]

# ╔═╡ f67cf7f4-a1d1-4e55-85b5-a11d966fabe9
plot_params_A_diff(results["MARX-WI"].envs_train[hyperparams_run_WI.run], results["MARX-WI"].agents[hyperparams_run_WI.run], psize=(4*dpi,0.6dpi), indices=A_indices, f_name="figures/$(label_env)-coeffs-A-diff.png")

# ╔═╡ 2aa3144e-936b-4ab9-bbca-088e65e1cb76
plot_param_M_timeseries(results["MARX-WI"].recs_train[hyperparams_run_WI.run,hyperparams_debug.T], sys=results["MARX-WI"].envs_train[hyperparams_run_WI.run,hyperparams_debug.T], indices=A_indices, f_name="figures/$(label_env)-coeffs-A-timeline-selection.png")

# ╔═╡ 744b7c90-ae12-4081-9475-d840879aedad
plot_ces_posterior_likelihood_kls_posterior_prior(results["MARX-WI"].recs_train[hyperparams_run_WI.run,hyperparams_debug.T]; show_surprisals=true, f_name="figures/$(label_env)-ces_posterior_likelihood_kls_posterior_prior.png")

# ╔═╡ 67af74c2-b8ae-4d0e-81a9-b835ab99cc0a
plot_es_posterior(results["MARX-WI"].recs_train[hyperparams_run_WI.run,hyperparams_debug.T], f_name="figures/$(label_env)-es_posterior.png")

# ╔═╡ a3ef187b-6773-4ee6-a56b-044ca28fd95e
md"## DEBUGGING: Inspecting performance of MARX-UI of a single run for T=$(hyperparams_debug.T)"

# ╔═╡ 37178881-e714-4c64-bbd7-7d4b321a5405
@bind hyperparams_run_UI MARX.ui_choices([
	Dict(:type => :Slider, :label => "run", :range => 1:N_runs, :default => 1),
])

# ╔═╡ 1e78cacd-0b65-4f00-a35b-0247260895d3
plot_param_W_timeseries(results["MARX-UI"].recs_train[hyperparams_run_UI.run,hyperparams_debug.T], sys=results["MARX-UI"].envs_train[hyperparams_run_UI.run,hyperparams_debug.T])

# ╔═╡ 24a3f77f-dadf-4706-94b9-b9a7dcf5b16b
plot_params_A_diff(results["MARX-UI"].envs_train[hyperparams_run_UI.run], results["MARX-UI"].agents[hyperparams_run_UI.run], psize=(4*dpi,0.6dpi), indices=A_indices)

# ╔═╡ ce5fac9b-e242-45f4-8552-ffbb3f457716
plot_param_M_timeseries(results["MARX-UI"].recs_train[hyperparams_run_UI.run,hyperparams_debug.T], sys=results["MARX-UI"].envs_train[hyperparams_run_UI.run,hyperparams_debug.T], indices=A_indices)

# ╔═╡ 6780990f-7775-4261-8b16-8eefb0b83294
plot_ces_posterior_likelihood_kls_posterior_prior(results["MARX-UI"].recs_train[hyperparams_run_UI.run,hyperparams_debug.T]; show_surprisals=true)

# ╔═╡ d1fb0657-d68c-4755-b40d-d6f96a478529
plot_es_posterior(results["MARX-UI"].recs_train[hyperparams_run_UI.run,hyperparams_debug.T])

# ╔═╡ Cell order:
# ╟─e389e050-65e2-11ef-3cae-03e1bfe027f0
# ╠═a3fd9a34-ed1a-4b78-aca0-acf7bfc99a7b
# ╠═b7debaa5-8ebd-4101-816c-9a44f49c40e9
# ╠═6899ed0a-d995-4c26-8a0c-a7a58b9d02e7
# ╠═636f758f-f182-4d11-b29c-ae2e46ef47f2
# ╠═99a54091-3da8-4964-9b36-7e4e501579d0
# ╠═897f2a85-1353-43e6-bdc6-6d1f2f07c1e1
# ╟─a0a31c51-2a1c-4251-adf8-2ed10272ac2c
# ╠═b45c953c-36cb-497d-9968-d139392bf441
# ╟─94836524-fdf7-4bcf-a65a-3343f724a196
# ╟─7b412a6c-a8a5-4c93-8083-11acbd1e5567
# ╟─e98d41f6-0d08-4677-b283-d589b7b9c2a5
# ╠═c99307a2-cfbf-471e-8a3b-0d4c694ea712
# ╠═9c9729a3-226b-493f-9ee8-4104ff6ff8c5
# ╠═2792d246-0377-40f7-adf5-ef99d023aa9f
# ╠═87614be1-7870-444d-b985-8dda3844aa2f
# ╠═408cb4a4-3e56-45e9-9b08-cf43116ceb18
# ╟─124996b5-8df5-4207-9950-ca700c8a0ade
# ╟─d88f5e6b-3afd-456f-a796-2b316cbe7cc3
# ╟─0bb97d07-7628-4973-8bf9-cbfb96438cfa
# ╠═ab00b2e0-b28a-4224-92d4-add711923ff1
# ╠═425cb22a-def8-4aa5-bb27-96b2ad0b9e09
# ╠═f67cf7f4-a1d1-4e55-85b5-a11d966fabe9
# ╠═2aa3144e-936b-4ab9-bbca-088e65e1cb76
# ╠═744b7c90-ae12-4081-9475-d840879aedad
# ╠═67af74c2-b8ae-4d0e-81a9-b835ab99cc0a
# ╟─a3ef187b-6773-4ee6-a56b-044ca28fd95e
# ╠═37178881-e714-4c64-bbd7-7d4b321a5405
# ╠═1e78cacd-0b65-4f00-a35b-0247260895d3
# ╠═24a3f77f-dadf-4706-94b9-b9a7dcf5b16b
# ╠═ce5fac9b-e242-45f4-8552-ffbb3f457716
# ╠═6780990f-7775-4261-8b16-8eefb0b83294
# ╠═d1fb0657-d68c-4755-b40d-d6f96a478529
