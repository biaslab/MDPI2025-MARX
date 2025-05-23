module Plotting

using Plots # make sure to ]add FFMPEG
#gr()
#plotly()
using LaTeXStrings
using Distributions # Normal
using Printf # @sprintf
using StatsPlots # covellipse!
using LinearAlgebra # diag
using ColorSchemes
using Statistics
using DataFrames
# palette = distinguishable_colors(D_y^2, ColorSchemes.colorblind8.colors)
#default(color_palette = ColorSchemes.colorblind8.colors)

using ..AgentBase
using ..MARXAgents
using ..SystemBase
using ..DynamicSystemBase
using ..MassSpringSystems
using ..MARXSystems
using ..Pendulums
using ..Utils
using ..InteractionRecorder

export plot_controls, plot_observations, plot_trial_test_dream
export plot_error
export plot_system, animate_system, plot_predictions, plot_monte_carlo_results, plot_agent_performance, plot_agent_params, plot_params
export plot_param_M_timeseries, plot_param_M_compare_timeseries
export plot_param_M, plot_param_M_combos, plot_update_M
export plot_param_A_norm, plot_param_W_norm, plot_param_AW_norm
export violinplot_param_A_norm, violinplot_param_W_norm, violinplot_param_AW_norm
export plot_pdf_params
export plots_paper, plots_paper_single
export plot_pdf_predictive
export plot_surprisals, plot_ces_posterior_likelihood, plot_ces_posterior_prior, plot_es_posterior
export plot_ces_posterior_likelihood_kls_posterior_prior, plot_ces_posterior_likelihood_ces_posterior_prior_es_posterior
export plot_system_params_A, plot_params_A_comparison, plot_params_A_diff
export plot_param_W_timeseries
export plot_param_Ω, plot_param_Ω_combos, plot_update_Ω, plot_param_Ω_timeseries
export plot_param_Λ, plot_param_Λ_combos, plot_update_Λ, plot_param_Λ_timeseries, plot_param_Λ_timeseries_dim
export plot_param_ΛΩ
export plot_system_params_W, plot_params_W_comparison
export plot_det_ΛΩ
export plot_rmse_boxplot, plot_rmse_line, plot_rmse_line_baseline, plot_rmse_violin
export plot_rmse_heatmap_Nys_Nus

"""    tickfontsize: Font size for the tick labels (x and y axis).
    guidefontsize: Font size for the axis labels (x and y labels).
    legendfontsize: Font size for the legend text.
    titlefontsize: Font size for the title.
    labelrotation: Rotation of the x and y axis labels (can be useful for adjusting space).
    series_annotations_fontsize: Font size for series annotations
        Title: 18-20 points
    Axis Labels: Title font size minus 2-4 points
    Tick Labels: Axis label font size minus 2-4 points
    Legend: Axis label font size or tick label font size

    Defaults for Plots:
    guidefontsize = 11
    titlefontsize = 10
    tickfontsize = 8
    legendfontsize = 8
    """

    """
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    """

titlefontsize=16
guidefontsize=titlefontsize-1
tickfontsize=titlefontsize-2
tickfontsize=titlefontsize
legendfontsize=guidefontsize

guidefontsize=15
titlefontsize=guidefontsize-1
tickfontsize=guidefontsize-3
legendfontsize=tickfontsize+1

DEFAULTS = (
    titlefontsize=titlefontsize,
    guidefontsize=guidefontsize,
    tickfontsize=tickfontsize,
    legendfontsize=tickfontsize,
    dpi=300,
    width_in=4,
    height_in=1,
    #width_in=5,
    #height_in=2,
    fillalpha=0.2,
    linewidth=2,
    linestyle=:dashdot
)

label_pdf_params_true = latexstring("\\log p(\\Theta = \\tilde{\\Theta} \\mid \\mathcal{D}_k)")
label_norm_A = latexstring("||\\tilde{A} - A||_F")
label_norm_A_log = latexstring("\\log(||\\tilde{A} - A||_F)")
label_norm_W = latexstring("||\\tilde{W} - W||_F")
label_norm_W_log = latexstring("\\log(||\\tilde{W} - W||_F)")
label_pdf_param_A = latexstring("\\log p(A \\mid W, \\mathcal{D}_k)")
label_pdf_param_W = latexstring("\\log p(W \\mid \\mathcal{D}_k)")
label_surprisals = latexstring("-\\log p(\\tilde{y}_k \\mid \\mathcal{D}_k)")
label_es_posterior = latexstring("H[q(\\Theta \\mid \\mathcal{D}_k)]")
label_ces_posterior_prior = latexstring("H[q(\\Theta \\mid \\mathcal{D}_k), p(\\Theta \\mid \\mathcal{D}_{k-1})]")
label_ces_posterior_likelihood = latexstring("H[q(\\Theta \\mid \\mathcal{D}_k), p(\\tilde{y}_k \\mid \\Theta, x_k)]")
label_kls_posterior_prior = latexstring("D_{KL}[q(\\Theta \\mid \\mathcal{D}_k) \\mid\\mid p(\\Theta \\mid \\mathcal{D}_{k-1})]")

# Helper function for initializing a plot with common properties
function initialize_plot(title::String, xlabel::String, ylabel::String, grid::Bool=true)
    return plot(
        grid=grid,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi),
        legend=:bottomright,
        tickfontsize=DEFAULTS.tickfontsize,
        guidefontsize=DEFAULTS.guidefontsize,
        legendfontsize=DEFAULTS.legendfontsize,
        titlefontsize=DEFAULTS.titlefontsize)
end


# Helper function to handle plot limits
function finalize_plot!(x_limits::Union{Tuple{Float64, Float64}, Nothing}, y_limits::Union{Tuple{Float64, Float64}, Nothing})
    if !isnothing(x_limits) xlims!(x_limits) end
    if !isnothing(y_limits) ylims!(y_limits) end
end

# Common plotting logic for observed, predicted, and goal positions (for carts)
function plot_carts!(rec::Recorder, sys::DoubleMassSpringDamperSystem, t::Int64, t_real::Int64, p_offset::Vector{Float64}, y_offsets::Vector{Float64}, scales::Vector{Float64}, opacities::Vector{Float64}, cart_colors::Vector{Symbol}, com_color::Symbol, gs::Union{Vector{Distributions.Normal{Float64}}, Nothing}, pys::Union{Vector{Float64}, Nothing}, vys::Union{Vector{Float64}, Nothing})
    x1, x2, dx1, dx2 = sys.zs[t_real]
    y1, y2 = rec.ys[:, t+1]

    # CART: observed position
    p_carts_observed = calculate_cart_positions([y1, y2], p_offset, y_offsets[1])
    plot_cart_box_and_com!(p_carts_observed, scales[1], scales[1], scales[1], y_offsets[1], opacities[1], cart_colors, com_color)

    # CART: desired/goal
    if !isnothing(gs)
        p_carts_goal = calculate_cart_positions(mean.(gs), p_offset, y_offsets[3])
        plot_distributions!(p_carts_goal, std.(gs), -0.25, -1, cart_colors)
    end

    # CART: predictions
    if !isnothing(pys) && !isnothing(vys)
        p_carts_pred = calculate_cart_positions(pys, p_offset, y_offsets[2])
        plot_distributions!(p_carts_pred, sqrt.(vys), 0.25, 1, cart_colors)
    end
end

# Common plotting logic for links (for pendulums)
function plot_links!(sys::DPendulum, θ1::Float64, θ2::Float64, l1::Float64, l2::Float64, colors::Vector{Symbol}, gs::Union{Vector{Distributions.Normal{Float64}}, Nothing}, pys::Union{Vector{Float64}, Nothing}, vys::Union{Vector{Float64}, Nothing}, n_std::Float64)
    p_origin = Point(0.0, 0.0)

    # LINK
    p_link_1 = p_origin + Point(polar2cart(θ1, l1)...)
    p_link_2 = p_link_1 + Point(polar2cart(θ2, l2)...)
    draw_line!([p_origin, p_link_1], label="upper link", color=colors[1], alpha=1.0)
    draw_line!([p_link_1, p_link_2], label="lower link", color=colors[2], alpha=1.0)

    # LINK: desired (goal)
    if !isnothing(gs)
        μg1 = mean.(gs)
        σgs = std.(gs)
        for i in LinRange(-n_std, n_std, 7)
            θ_goal, r_goal = cart2polar(μg1...)
            x_goal, y_goal = polar2cart(θ_goal + i * σgs[1] - Float64(π)/2, r_goal)
            p_link_1 = p_origin + Point(x_goal, y_goal)
            draw_line!([p_origin, p_link_1], color=:green, alpha=0.15)
        end
    end

    # LINK: prediction
    if !isnothing(pys) && !isnothing(vys)
        σys = sqrt.(vys)
        for i in LinRange(-n_std, n_std, 7)
            p_link_1 = p_origin + Point(polar2cart(pys[1] + i * σys[1], l1)...)
            p_link_2 = p_link_1 + Point(polar2cart(pys[2] + i * σys[2], l2)...)
            draw_line!([p_origin, p_link_1], color=colors[1], alpha=0.15)
            draw_line!([p_link_1, p_link_2], color=colors[2], alpha=0.15)
        end
    end
end

function plot_system(rec::Recorder, sys::DoubleMassSpringDamperSystem, t::Int64;
    box_l::Float64=1.0,
    p_offset::Vector{Float64}=[0.0, 0.0],
    gs::Union{Vector{Distributions.Normal{Float64}}, Nothing}=nothing,
    #pys::Union{Vector{Float64}, Nothing}=nothing,
    #vys::Union{Vector{Float64}, Nothing}=nothing,
    y_offsets::Vector{Float64}=[0.0, 0.5, -0.5], # offset: observation, prediction, goal
    scales::Vector{Float64}=[0.5, 0.5, 0.5], # scale: observation, prediction, goal
    opacities::Vector{Float64}=[1.0, 0.05, 0.05], # opacity: observation, prediction, goal
    cart_colors::Vector{Symbol}=[:orange, :blue],
    com_color::Symbol=:black,
    x_limits::Tuple{Float64, Float64}=(-2.0, 4.0),
    y_limits::Tuple{Float64, Float64}=(-1.0, 1.0),
    xlabel::String="displacement",
    ylabel::String="height",
    prefix_title::String="")

    t_real = t
    plt = initialize_plot(@sprintf("%st = %.1f", prefix_title, t), xlabel, ylabel)
    plot_carts!(rec, sys, t, t_real, p_offset, y_offsets, scales, opacities, cart_colors, com_color, gs, rec.pred_ys[:,t], diag(rec.pred_Σs[:,:,t]))
    finalize_plot!(nothing, y_limits)
    return plt
end

# Refactored plot_system for DPendulum
function plot_system(rec::Recorder, sys::DPendulum, t::Int64; box_l::Float64=1.0, gs::Union{Vector{Distributions.Normal{Float64}}, Nothing}=nothing, pys::Union{Vector{Float64}, Nothing}=nothing, vys::Union{Vector{Float64}, Nothing}=nothing, n_std::Float64=1.0, colors::Vector{Symbol}=[:orange, :blue], prefix_title::String="", f_name::Union{Nothing, String}=nothing)
    θ1, θ2 = rec.ys[:,t]
    l1, l2 = sys.length
    lims = 1.1 * (l1 + l2)

    plt = initialize_plot("$prefix_title t = $t", "x", "y")
    plot_links!(sys, θ1, θ2, l1, l2, colors, gs, pys, vys, n_std)
    finalize_plot!((-lims, lims), (-lims, lims))
    plot!(size=(600,600)) # force size to be quadratic

    if !isnothing(f_name) savefig(f_name) end

    return plt
end

function draw_line!(ps::Vector{Point}; linewidth=10, alpha=0.6, color="black", label="")
    (xs, ys) = get_values(ps)
    return Plots.plot!(xs, ys, linewidth=linewidth, alpha=alpha, color=color, label=label)
end

function plot_cart_box_and_com!(p_carts::Vector{Vector{Float64}}, box_w::Float64, box_h::Float64, scale::Float64, y_offset::Float64, opacity::Float64, color::Vector{Symbol}, com_color::Symbol)
    for (cart_idx, p_cart) in enumerate(p_carts)
        plot!(
            rectangle(scale * box_w, scale * box_h, p_cart[1] - scale * 0.5 * box_w, p_cart[2] - scale * 0.5 * box_h),
            opacity = opacity, color = color[cart_idx], label = nothing
        )
        scatter!([p_cart[1]], [p_cart[2] + y_offset], color = com_color, label = nothing, opacity = opacity)
    end
end

function plot_distributions!(p_carts::Vector{Vector{Float64}}, σ_values::Vector{Float64}, y_offset_base::Float64, y_direction::Int64, cart_colors::Vector{Symbol})
    for (cart_idx, p_cart) in enumerate(p_carts)
        x = p_cart[1] - 2:0.01:p_cart[1] + 2
        dist = Normal(p_cart[1], σ_values[cart_idx])
        y_offset = y_offset_base .+ y_direction * pdf(dist, x)
        plot!(x, y_offset, fillalpha=DEFAULS.fillalpha, fillrange=y_offset_base, color=cart_colors[cart_idx])
        plot!([p_cart[1], p_cart[1]], [y_offset_base, y_offset_base .+ y_direction * pdf(dist, p_cart[1])], linestyle=:dash, color=cart_colors[cart_idx], linewidth=2)
    end
end

function animate_system(f_name::String, sys::DoubleMassSpringDamperSystem; fps::Int=60, prefix_title::String="")
    anim = @animate for k in 1:sys.N
        plot_system(rec, sys, k, prefix_title=prefix_title)
    end
    return mp4(anim, f_name, fps=fps, loop=0)
end

function plot_controls(rec::Recorder, sys::System; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    plots = []
    xlabel = "time [s]"
    ylabel_base = "torque"
    xticks_pos, xticks_labels = get_realtime_xticks(sys)
    (D_u, _) = size(rec.us)
    (D_a, N) = size(rec.as)
    for i in 1:D_a
        title = "controls/actions"
        push!(plots, plot(legend=nothing))
        if i == 1
            title!(title)
            plot!(legend=:topleft)
        end
        if i == 2
            xlabel!(xlabel)
        end
        ylabel = i == 1 ? "$(ylabel_base)₁" : "$(ylabel_base)₂"
        ylabel!(ylabel)
        plot!(rec.as[i, :], color="purple", label="actions")
        plot!(rec.us[i, :], color="red", label="controls")
    end
    p = plot(plots..., layout=grid(2,1))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_observations(rec::Recorder, sys::System; probabilistic::Bool=true)
    tsteps = range(0, step=sys.Δt, length=sys.N)
    D_y = get_observation_dim(typeof(sys))
    σs = probabilistic ? hcat([ sqrt.(diag(rec.pred_Σs[:,:,t,1])) for t in 1:sys.N ]...) : nothing
    plots = []
    for i in 1:D_y
        push!(plots, plot(tsteps, rec.ys[i,:], labels=latexstring("y_{$(i)}"), xlabel="time", ylabel="signal amplitude", legend=:topleft))
        plot!(tsteps, rec.pred_ys[i,:], ribbon=probabilistic ? σs[i,:] : nothing, labels=latexstring("\\hat{y}_{$(i)}"))
    end

    p = plot(plots..., layout=(D_y,1), size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_trial_test_dream(rec_test::Recorder, rec_test_dream::Recorder, env_test_dream::System; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    labels_pred = ["ŷ₁", "ŷ₂"]
    labels_true = ["y₁", "y₂"]
    t_beg_test = 1
    t_end_test = env_test_dream.N
    t_beg_dream = 1
    t_end_dream = env_test_dream.N
    σs_test = hcat([ sqrt.(diag(rec_test.pred_Σs[:,:,t,1])) for t in t_beg_test:t_end_test ]...)
σs_dream = hcat([ sqrt.(diag(rec_test_dream.pred_Σs[:,:,t,1])) for t in t_beg_dream:t_end_dream ]...)
    plots = []
    for dim in 1:2
        pys = vcat(rec_test.pred_ys[dim,t_beg_test:t_end_test], rec_test_dream.pred_ys[dim,t_beg_dream:t_end_dream])
        tys = vcat(rec_test.ys[dim,t_beg_test:t_end_test], rec_test_dream.ys[dim,t_beg_dream:t_end_dream])
        σs = hcat(σs_test, σs_dream)
        push!(plots, plot(pys, ribbon=σs[dim,:], label=labels_pred[dim]))
        plot!(tys, label=nothing)
        scatter!(tys, markershape=:+, label=labels_true[dim])
        vline!([t_end_test], linestyle=:dot, color=:blue, linewidth=2, label=nothing)
        if dim != 2
            plot!(xticks = nothing)
        end
        if dim == 2 xlabel!("time [s]") end
 #plot!([x_pos, x_pos], [0, 5], linestyle = :dot, color = :blue, linewidth = 2, label = "")
    end
    p = plot(plots..., layout=(length(plots), 1))
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(bottom_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_error(rec::Recorder, sys::System; f_name::Union{Nothing, String}=nothing)
    error = rec.ys .- rec.pred_ys[:,:,1]
    abs_error = abs.(error)
    (D_y, N) = size(rec.ys)
    rmse = [ sqrt(mean(error[dim,:].^2)) for dim in 1:D_y ]
    tsteps = range(0, step=sys.Δt, length=sys.N)
    plots = []
    for dim in 1:D_y
        push!(plots, plot(tsteps, abs_error[dim,:], label="RMSE = $(rmse[dim])", xlabel="time [s]", ylabel=latexstring("|y_{$(dim)} - \\hat{y}_{$(dim)}|")))
    end
    n_plots = length(plots)
    p = plot(plots..., layout=(n_plots,1), size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_predictions(rec::Recorder, sys::System, t_beg::Int, t_step::Int; lim_offset::Float64=0.5, t_prev::Int=0)
    plots = []
    push!(plots, plot_predictions_single(rec, sys, t_beg, t_step; focus_predictions=false, lim_offset=lim_offset, t_prev=t_prev))
    push!(plots, plot_predictions_single(rec, sys, t_beg, t_step; focus_predictions=true, lim_offset=lim_offset, t_prev=t_prev))
    p = plot(plots..., layout=(1,2))
    t_end = t_beg + t_step
    plot!(title="$t_beg - $t_end")
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_predictions_single(rec::Recorder, sys::System, t_beg::Int, t_step::Int; focus_predictions::Bool=false, lim_offset::Float64=0.5, t_prev::Int=0)
    t_end = t_beg + t_step
    ys_origin = rec.ys[:,t_beg:t_beg+1]
    ys = rec.ys[:, t_beg:t_end]
    pred_ys = rec.pred_ys[:, t_beg:t_end]
    pred_Σs = rec.pred_Σs[:, :, t_beg:t_end]
    p = plot()
    #scatter!([0.0], [0.0], label="origin", color=:red, markershape=:rect)
    if t_prev > 0
        color_past = :gray
        ys_past = rec.ys[:, t_beg - t_prev : t_beg-1]
        plot!(ys_past[1,:], ys_past[2,:], color=color_past, label=L"y_{<}")
        scatter!(ys_past[1,:], ys_past[2,:], label=nothing, color=color_past, markershape=:diamond)
    end

    #scatter!(ys_origin[1,:], ys_origin[2,:], label=nothing, color=:black, markershape=:diamond)

    plot!(ys[1,:], ys[2,:], color=:orange, label=L"y")
    scatter!(ys[1,:], ys[2,:], label=nothing, color=:orange, markershape=:diamond)

    plot!(pred_ys[1,:], pred_ys[2,:], color=:purple, label=L"\hat{y}")
    scatter!(pred_ys[1,:], pred_ys[2,:], label=nothing, color=:purple, markershape=:circle)
    for t in 1:t_step+1
        covellipse!(pred_ys[:,t], pred_Σs[:,:,t], n_std=1, aspect_ratio=1.0, label=false, color=:blue, showaxes=true, fillalpha=DEFAULTS.fillalpha, alpha=0.1)
    end
    xlim_ys = focus_predictions ? pred_ys[1,:] : ys[1,:]
    ylim_ys = focus_predictions ? pred_ys[2,:] : ys[2,:]
    xlim = [minimum(xlim_ys), maximum(xlim_ys)] + [-lim_offset, lim_offset]
    ylim = [minimum(ylim_ys), maximum(ylim_ys)] + [-lim_offset, lim_offset]
    xlims!(xlim...)
    ylims!(ylim...)
    return p
end

function plot_monte_carlo_results(f_name::String, abs_errors::Array{Float64, T}) where T
    D_y, N_interaction, N_experiments = size(abs_errors)
    plots = []
    for i in 1:D_y
        subp = plot()
        plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
        plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
        mean_pred_y = mean.(abs_errors[i,:,:])
        scatter!([N_interaction], [mean_pred_y], label=nothing)
        push!(plots, subp)
    end
    p = plot(plots..., layout=(2,1))
    savefig(f_name)
    return p
end

function plot_agent_performance(f_name::String, abs_errors_all::Vector{Array{Float64, T}}) where T
    N_e = length(abs_errors_all)
    D_y, _, _ = size(abs_errors_all[1])
    plots = []
    for i in 1:D_y
        subp = plot()
        plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
        plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
        Ns = zeros(N_e)
        σs = zeros(N_e)
        μs = zeros(N_e)

        for (j, abs_errors) in enumerate(abs_errors_all)
            _, N_interaction, _ = size(abs_errors)
            Ns[j] = N_interaction
            xs = [mean(row) for row in eachrow(abs_errors[i,:,:])]
            μs[j] = mean(xs)
            σs[j] = std(xs)
        end
        plot!(Ns, μs, ribbon=σs, label=nothing, xscale=:log2)
        push!(plots, subp)
    end
    p = plot(plots..., layout=(2,1))
    savefig(f_name)
    return p
end

function plot_agent_params(f_name::String, agent::MARXAgent)
    plots = []
    push!(plots, heatmap(agent.U'[:,:], title="U"))
    push!(plots, heatmap(agent.V'[:,:], title="V"))
    push!(plots, heatmap(agent.M'[:,1:agent.N_y*agent.D_y], title="M_y"))
    if agent.N_u > 0 push!(plots, heatmap(agent.M'[:,agent.N_y*agent.D_y+1:end], title="M_u")) end
    p = plot(plots..., layout=(2,2))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    savefig(f_name)
    return p
end

function plot_params(rec::Recorder, agent::MARXAgent, t::Int)
    Λs = rec.Λs[:,:,t]
    Ωs = rec.Ωs[:,:,t]
    Ms = rec.Ms[:,:,t]
    plots = []
    push!(plots, heatmap(Λs, title="Λ"))
    push!(plots, heatmap(Ωs, title="Ω"))
    push!(plots, heatmap(Ms[:,1:agent.N_y*agent.D_y], title="M_y"))
    if agent.N_u > 0 push!(plots, heatmap(Ms[:,agent.N_y*agent.D_y+1:end], title="M_u")) end
    p = plot(plots..., layout=(2,2))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_params_A_comparison(rec::Recorder, agent::Agent, sys::System, t::Int; f_name::Union{Nothing,String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    D_y = get_observation_dim(typeof(sys))
    D_x = sys.D_x
    yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        yticks = (1:D_y, ["yₖ,₁", "yₖ,₂"])
    end
    ticks = get_memory_ticks(sys.memory_type, D_y, agent.N_y, agent.N_u)
    xticks = (1:D_x, ticks)
    A_true = sys.A
    A_pred = rec.Ms[:,:,t]
    clims = (
        min(minimum(A_true), minimum(A_pred)),
        max(maximum(A_true), maximum(A_pred))
    )
    fillalpha = 0.75
    hmcolor = :viridis
    plots = []
    push!(plots, heatmap(A_true, clims=clims, yticks=1:size(sys.A, 1), yflip=true, xticks=nothing, color=hmcolor, fillalpha=fillalpha))
    for (i,j) in [(1,1), (1,2), (1,3), (1,4), (1,5), (2,6), (2,7), (2,8), (2,9), (2,10)]
    #for i in 1:D_y, j in 1:D_x
        annotation_text = "X"
        annotate!(j, i, text(annotation_text, :black, 14, :center))
    end
    push!(plots, heatmap(A_pred, clims=clims, yticks=1:size(sys.A, 1), yflip=true, xticks=nothing, color=hmcolor, fillalpha=fillalpha))
    push!(plots, heatmap(A_true .- A_pred, yticks=1:size(sys.A, 1), yflip=true, xticks=xticks, color=hmcolor, fillalpha=fillalpha))
    n_plots = length(plots)
    p = plot(plots..., layout=(n_plots,1))
    #plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(bottom_margin=8Plots.mm)
    plot!(yticks=yticks)
    plot!(link=:x)
    #plot!(tickfontsize = 18, colorbar_tickfontsize=10)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_params_A_diff(sys::System, agent::Agent; indices::Vector{Tuple{Int64,Int64}}=[], f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, tickfontsize::Union{Nothing, Int}=nothing)
    D_y = get_observation_dim(typeof(sys))
    D_x = sys.D_x
    xticks = (1:D_x, latexstring("x_{$(i)}") for i in 1:D_x) # [ latexstring("x_{$(i)}") for i in 1:D_x ]
    yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        yticks = (1:D_y, ["yₖ,₁", "yₖ,₂"])
    end

    ticks = get_memory_ticks(sys.memory_type, D_y, agent.N_y, agent.N_u)
    xticks = (1:D_x, ticks)

    fillalpha = 0.8
    hmcolor = :viridis
    #plots = []
    #push!(plots, heatmap(A_true, clims=clims, yticks=1:size(sys.A, 1), yflip=true, xticks=nothing, color=hmcolor, fillalpha=fillalpha))
    p = heatmap(sys.A .- agent.M, yticks=1:size(sys.A, 1), yflip=true, color=hmcolor, fillalpha=fillalpha)
    for (i,j) in indices # [(1,1), (1,2), (1,3), (1,4), (1,5), (2,6), (2,7), (2,8), (2,9), (2,10)]
    #for i in 1:D_y, j in 1:D_x
        annotation_text = "X"
        annotate!(j, i, text(annotation_text, :black, 14, :center))
    end

    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm, right_margin=8Plots.mm)

    plot!(xticks=xticks, yticks=yticks)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_system_params_A(sys::System, agent::Agent; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, tickfontsize::Union{Nothing, Int}=nothing)
    D_y = get_observation_dim(typeof(sys))
    D_x = sys.D_x
    xticks = (1:D_x, latexstring("x_{$(i)}") for i in 1:D_x) # [ latexstring("x_{$(i)}") for i in 1:D_x ]
    yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        yticks = (1:D_y, ["yₖ,₁", "yₖ,₂"])
    end

    ticks = get_memory_ticks(sys.memory_type, D_y, agent.N_y, agent.N_u)
    xticks = (1:D_x, ticks)

    fillalpha = 0.8
    hmcolor = :viridis
    #plots = []
    #push!(plots, heatmap(A_true, clims=clims, yticks=1:size(sys.A, 1), yflip=true, xticks=nothing, color=hmcolor, fillalpha=fillalpha))
    p = heatmap(sys.A, yticks=1:size(sys.A, 1), yflip=true, color=hmcolor, fillalpha=fillalpha)
    for (i,j) in [(1,1), (1,2), (1,3), (1,4), (1,5), (2,6), (2,7), (2,8), (2,9), (2,10)]
    #for i in 1:D_y, j in 1:D_x
        annotation_text = "X"
        annotate!(j, i, text(annotation_text, :black, 14, :center))
    end

    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm, right_margin=8Plots.mm)

    plot!(xticks=xticks, yticks=yticks)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_system_params_W(sys::System; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, tickfontsize::Union{Nothing, Int}=nothing)
    D_y = get_observation_dim(typeof(sys))
    D_x = sys.D_x
    xticks = yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        xticks = yticks = (1:D_y, ["yₜ,₁", "yₜ,₂"])
    end
    p = heatmap(sys.W, yticks=1:size(sys.A, 1), yflip=true)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    #plot!(bottom_margin=4Plots.mm)
    plot!(xticks=xticks, yticks=yticks)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_params_W_comparison(rec::Recorder, sys::System, t::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, tickfontsize::Union{Nothing, Int}=nothing)
    D_y = get_observation_dim(typeof(sys))
    xticks = yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        xticks = yticks = (1:D_y, ["yₜ,₁", "yₜ,₂"])
    end
    epsilon = 1e-10
    W_true = sys.W
    #W_true = log.(W_true .+ epsilon)
    W_pred = rec.Ws[:,:,t]
    #W_pred = log.(W_pred .+ epsilon)
    clims = (
        min(minimum(W_true), minimum(W_pred)),
        max(maximum(W_true), maximum(W_pred))
    )
    plots = []
    push!(plots, heatmap(W_true, clims=clims, yflip=true, xticks=xticks, yticks=yticks, colorbar=false))
    push!(plots, heatmap(W_pred, clims=clims, yflip=true, xticks=xticks, yticks=nothing, colorbar=true))
    push!(plots, heatmap(W_true .- W_pred, yflip=true, xticks=xticks, yticks=nothing, colorbar=true))
    push!(plots, heatmap(rec.Ωs[:,:,t], yflip=true, xticks=xticks, yticks=nothing, colorbar=true))
    n_plots = length(plots)
    p = plot(plots..., layout=(2, 2))
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    #plot!(left_margin=8Plots.mm)
    #plot!(right_margin=8Plots.mm)
    plot!(bottom_margin=8Plots.mm)
    #plot!(xticks=xticks, yticks=yticks)
    if !isnothing(f_name) savefig(f_name) end
    #x = heatmap(W_true, clims=(0,1), framestyle=:none, lims=(-1,0), colorbar=true)
    #if !isnothing(f_name) savefig("$(f_name)_cbar.png") end
    return p
end

function plot_param_M_timeseries(
    rec::Recorder; 
    sys::Union{Nothing, System}=nothing, 
    f_name::Union{Nothing, String}=nothing, 
    psize::Union{Nothing, Tuple}=nothing, 
    y_range::Union{Nothing, UnitRange{Int64}}=nothing, 
    x_range::Union{Nothing, UnitRange{Int64}}=nothing, 
    indices::Union{Nothing, Vector{Tuple{Int64, Int64}}}=nothing, 
    ylim::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    label = label_pdf_param_A
    (D_y, D_x, N) = size(rec.Ms)

    Σs = zeros(D_y*D_x, D_y*D_x, N)
    std_A = zeros(D_y, D_x, N)

    for t in 1:N
        Σs[:,:,t] = kron(inv(rec.Λs[:,:,t]), inv(rec.Ws[:,:,t]))
        std_A[:,:,t] = reshape(sqrt.(diag(Σs[:,:,t])), (D_y, D_x))
    end

    palette = theme_palette(:auto)
    #y_range = isnothing(y_range) ? 1:D_y : y_range
    #x_range = isnothing(x_range) ? 1:D_x : x_range
    y_range = 1:D_y
    x_range = 1:D_x

    p = plot(
        xlabel="time [s]",
        ylabel=label,
        legend=:top,
        legend_columns=4, # <<== Wide legend here
        legend_border=false,
        #legendtitle="Entries (i,j)",
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        bottom_margin=8Plots.mm,
        left_margin=8Plots.mm
    )

    subscripts_digit = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    cidx = 1
    if !isnothing(indices)
        for (i, j) in indices
            color = palette[(cidx - 1) % length(palette) + 1]
            label_appendix = "$(subscripts_digit[i]),$(subscripts_digit[j])"
            label_text = "ã$label_appendix"
            plot!(rec.Ms[i, j, :], ribbon=std_A[i, j, :], label=label_text, fillalpha=DEFAULTS.fillalpha, color=color, lw=DEFAULTS.linewidth)
            if !isnothing(sys) plot!(rec.As[i, j, :], label=nothing, color=color, lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle) end
            cidx += 1
        end
    else
        for (i, j) in Iterators.product(y_range, x_range)
            color = palette[(cidx - 1) % length(palette) + 1]
            label_appendix = "$i,$j"
            label_text = "ã$label_appendix"
            plot!(rec.Ms[i, j, :], ribbon=std_A[i, j, :], label=label_text, fillalpha=DEFAULTS.fillalpha, color=color, lw=DEFAULTS.linewidth)
            if !isnothing(sys) plot!(rec.As[i, j, :], label=nothing, color=color, lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle) end
            cidx += 1
        end
    end

    # Compute y-limits
    if isnothing(ylim)
        min_y = minimum(rec.Ms)
        max_y = maximum(rec.Ms)
        if !isnothing(sys)
            min_y = min(min_y, minimum(sys.A))
            max_y = max(max_y, maximum(sys.A))
        end
        y_limits = (min_y - 0.1, max_y + 0.1)
    else
        y_limits = ylim
    end

    ylims!(y_limits)

    if !isnothing(f_name)
        savefig(f_name)
    end

    return p
end

function plot_param_M_compare_timeseries(rec::Recorder, sys::System, dim_y::Int)
    p = plot(rec.Ms[dim_y,:,:]')
    hline!(sys.A[dim_y,:])
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(legend=:bottomright)
    return p
end

function plot_param_M(rec::Recorder, t::Int; f_name::Union{Nothing,String}=nothing, psize::Union{Nothing, Tuple}=nothing, xticks::Union{Nothing, Vector{String}}=nothing, yticks::Union{Nothing, Vector{String}}=nothing)
    (D_y, D_x, N) = size(rec.Ms)
    if isnothing(xticks)
        xticks = [latexstring("x_{$(i)}") for i in 1:D_x]
        xticks = ["x₁", "x₂", "x₃", "x₄", "x₅", "x₆", "x₇", "x₈", "x₉", "x₁₀"]
    end
    if isnothing(yticks)
        yticks = [latexstring("y_{$(i)}") for i in 1:D_y] # [ latexstring("y_{$(i)}") for i in 1:D_y ]
        #yticks = (1:D_y, ["yₜ,₁", "yₜ,₂"])
    end
    xticks = (1:D_x, xticks)
    yticks = (1:D_y, yticks)
    plots = []
    p = heatmap(rec.Ms[:,:,t], xticks=xticks, yticks=yticks, yflip=true, xrotation=-45)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(bottom_margin=12Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_A_norm(rec::Recorder; sys::Union{Nothing, System}=nothing, f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_y, D_x, N) = size(rec.Ms)
    palette = theme_palette(:auto)
    color = palette[1]
    p = plot([ norm(rec.Ms[:,:,t], 2) for t in 1:N ], label=latexstring("||A||_F"), color=color, linewidth=DEFAULTS.linewidth, xlabel="time [s]")
    if !isnothing(sys)
        hline!([norm(sys.A, 2)], lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle, label=latexstring("||\\tilde{A}||_F"), color=color)
    end
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_A_norm(recs::Matrix{Recorder}, syss::Matrix{MARXSystem}, T::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    N_runs, n_Ts = size(recs)
    (D_y, D_x, N) = size(recs[1,T].Ms)

    p = plot(xlabel="time [s]")
    palette = theme_palette(:auto)
    color = palette[1]
    norms = zeros(N_runs, N)
    for i in 1:N_runs
        A_diff = syss[i, T].A .- recs[i, T].Ms
        norms[i,:] = [ norm(A_diff[:,:,t],2) for t in 1:N ]
    end
    plot!(mean(norms, dims=1)', ribbon=std(norms, dims=1), label=latexstring("||A||_F"), color=color, linewidth=DEFAULTS.linewidth)
    hline!([0.0], label=nothing, lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function compute_A_norms(vrecs::Vector{Recorder}, vsys::Vector{MARXSystem}; logscale::Bool=false)
    N_runs = length(vrecs)
    (D_y, D_x, N) = size(vrecs[1].Ms)
    norms = zeros(N_runs, N)
    for i in 1:N_runs
        A_diff = vsys[i].A .- vrecs[i].Ms
        norms[i, :] .= [norm(A_diff[:, :, t], 2) for t in 1:N]
    end
    return logscale ? log.(norms) : norms
end

function compute_W_norms(vrecs::Vector{Recorder}, vsys::Vector{S}; logscale::Bool=false) where {S<:System}
    N_runs = length(vrecs)
    (D_y, D_x, N) = size(vrecs[1].Ws)
    norms = zeros(N_runs, N)
    for i in 1:N_runs
        W_diff = vsys[i].W .- vrecs[i].Ws
        norms[i, :] .= [norm(W_diff[:, :, t], 2) for t in 1:N]
    end
    return logscale ? log.(norms) : norms
end

function norms_to_dataframe(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; logscale=false)
    df = DataFrame(Label=String[], Time=Int[], Norm=Float64[])
    for (label, vrecs, vsys) in data
        norms = compute_A_norms(vrecs, vsys; logscale=logscale)
        N = size(norms, 2)
        mean_norm = vec(mean(norms, dims=1))
        std_norm = vec(std(norms, dims=1))
        for t in 1:N
            push!(df, (label, t, mean_norm[t]))
        end
    end
    return df
end

function norms_to_dataframe_A(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; logscale=false)
    df = DataFrame(Label=String[], Time=Int[], Norm=Float64[])
    for (label, vrecs, vsys) in data
        norms = compute_A_norms(vrecs, vsys; logscale=logscale)
        (N_runs, N_times) = size(norms)
        for t in 1:N_times
            for r in 1:N_runs
                push!(df, (label, t, norms[r, t]))
            end
        end
    end
    return df
end

function norms_to_dataframe_W(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; logscale=false)
    df = DataFrame(Label=String[], Time=Int[], Norm=Float64[])
    for (label, vrecs, vsys) in data
        if label == "RLS" continue end
        norms = compute_W_norms(vrecs, vsys; logscale=logscale)
        (N_runs, N_times) = size(norms)
        for t in 1:N_times
            for r in 1:N_runs
                push!(df, (label, t, norms[r, t]))
            end
        end
    end
    return df
end

function norms_to_dataframe_AW(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; logscale=false)
    df = DataFrame(Label=String[], Time=Int[], Norm_A=Float64[], Norm_W=Float64)
    for (label, vrecs, vsys) in data
        norms_A = compute_W_norms(vrecs, vsys; logscale=logscale)
        norms_W = compute_W_norms(vrecs, vsys; logscale=logscale)
        (N_runs, N_times) = size(norms)
        for t in 1:N_times
            for r in 1:N_runs
                push!(df, (label, t, norms_A[r, t], norms_W[r, t]))
            end
        end
    end
    return df
end

function violinplot_param_AW_norm(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, logscale::Bool=true, plot_trends::Bool=false)
    label_A = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    label_W = latexstring(logscale ? "\\log(||\\tilde{W} - W||_F)" : "||\\tilde{W} - W||_F")
    df_A = norms_to_dataframe_A(data; logscale=logscale)
    df_W = norms_to_dataframe_W(data; logscale=logscale)
    plots = []
    #plt_A = @df df_A violin(
    plt_A = @df df_A violin(
        :Time,
        :Norm,
        group=:Label,
        fillalpha=0.4,
        linewidth=0.8,
        legend=:topleft,
        #xlabel="time [s]",
        ylabel=label_A,
        palette=:tab10
    )
    push!(plots, plt_A)

    if plot_trends
        grouped = groupby(df, [:Label, :Time])
        mean_df = combine(grouped, :Norm => mean => :MeanNorm)

        for label in unique(df.Label)
            subdf = mean_df[mean_df.Label .== label, :]
            plot!(subdf.Time, subdf.MeanNorm, lw=2, marker=:circle, label=label)
        end
    end

    plt_W = @df df_W violin(
        :Time,
        :Norm,
        group=:Label,
        fillalpha=0.4,
        linewidth=0.8,
        legend=:topleft,
        xlabel="time [s]",
        ylabel=label_W,
        palette=:tab10,
        label=""
    )
    push!(plots, plt_W)

    if plot_trends
        grouped = groupby(df, [:Label, :Time])
        mean_df = combine(grouped, :Norm => mean => :MeanNorm)

        for label in unique(df.Label)
            subdf = mean_df[mean_df.Label .== label, :]
            plot!(subdf.Time, subdf.MeanNorm, lw=2, marker=:circle, label=label)
        end
    end

    plt = plot(plots..., layout=(2,1), link=:x)

    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end

function violinplot_param_A_norm(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, logscale::Bool=true, plot_trends::Bool=false)
    label_prefix = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    df = norms_to_dataframe_A(data; logscale=logscale)
    plt = @df df violin(
        :Time,
        :Norm,
        group=:Label,
        fillalpha=0.4,
        linewidth=0.8,
        legend=:topleft,
        xlabel="time [s]",
        ylabel=label_prefix,
        palette=:tab10
    )

    if plot_trends
        grouped = groupby(df, [:Label, :Time])
        mean_df = combine(grouped, :Norm => mean => :MeanNorm)

        for label in unique(df.Label)
            subdf = mean_df[mean_df.Label .== label, :]
            plot!(subdf.Time, subdf.MeanNorm, lw=2, marker=:circle, label=label)
        end
    end

    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end

function violinplot_param_W_norm(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, logscale::Bool=true, plot_trends::Bool=false)
    label_prefix = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    df = norms_to_dataframe_W(data; logscale=logscale)
    plt = @df df violin(
        :Time,
        :Norm,
        group=:Label,
        fillalpha=0.4,
        linewidth=0.8,
        legend=:topleft,
        xlabel="time [s]",
        ylabel=label_prefix,
        palette=:tab10
    )

    if plot_trends
        grouped = groupby(df, [:Label, :Time])
        mean_df = combine(grouped, :Norm => mean => :MeanNorm)

        for label in unique(df.Label)
            subdf = mean_df[mean_df.Label .== label, :]
            plot!(subdf.Time, subdf.MeanNorm, lw=2, marker=:circle, label=label)
        end
    end

    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end

function plot_param_A_norm(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, logscale::Bool=false)
    label_prefix = logscale ? label_norm_A_log : label_norm_A
    p = plot(xlabel="time [s]", ylabel=label_prefix)

    for (label, vrecs, vsys) in data
        norms = compute_A_norms(vrecs, vsys; logscale=logscale)
        mean_norm = vec(mean(norms, dims=1))
        std_norm = vec(std(norms, dims=1))
        plot!(mean_norm, ribbon=std_norm, lw=DEFAULTS.linewidth, label=label)
    end

    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end
    return p
end

function plot_param_W_norm(data::Vector{Tuple{String, Vector{Recorder}, Vector{S}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, logscale::Bool=false) where {S <: System}
    label_prefix = logscale ? label_norm_W_log : label_norm_W
    p = plot(xlabel="time [s]", ylabel=label_prefix)

    for (label, vrecs, vsys) in data
        norms = compute_W_norms(vrecs, vsys; logscale=logscale)
        mean_norm = vec(mean(norms, dims=1))
        std_norm = vec(std(norms, dims=1))
        plot!(mean_norm, ribbon=std_norm, lw=DEFAULTS.linewidth, label=label)
    end

    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end
    return p
end

function plot_param_AW_norm(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}};
    f_name::Union{Nothing, String}=nothing,
    psize::Union{Nothing, Tuple}=nothing,
    logscale::Bool=false
)
    # Define labels
    label_A = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    label_W = latexstring(logscale ? "\\log(||\\tilde{W} - W||_F)" : "||\\tilde{W} - W||_F")

    # Convert norms into DataFrames
    df_A = norms_to_dataframe_A(data; logscale=logscale)
    df_W = norms_to_dataframe_W(data; logscale=logscale)

    # --- Compute mean and std ---
    grouped_A = groupby(df_A, [:Label, :Time])
    meanstd_A = combine(grouped_A,
        :Norm => mean => :MeanNorm,
        :Norm => std => :StdNorm
    )

    grouped_W = groupby(df_W, [:Label, :Time])
    meanstd_W = combine(grouped_W,
        :Norm => mean => :MeanNorm,
        :Norm => std => :StdNorm
    )

    # --- Plot A ---
    p_A = @df meanstd_A plot(
        :Time,
        :MeanNorm,
        ribbon=:StdNorm,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        ylabel=label_A,
        legend=:topright,
        #xlabel=false,
        palette=:tab10
    )

    # --- Plot W ---
    p_W = @df meanstd_W plot(
        :Time,
        :MeanNorm,
        ribbon=:StdNorm,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        xlabel="time [s]",
        ylabel=label_W,
        legend=nothing,
        palette=:tab10
    )

    # --- Combine plots ---
    plt = plot(
        p_A, p_W,
        layout=(2,1),
        link=:x
    )

    # --- Style ---
    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=6Plots.mm,
        left_margin=8Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end


function plot_param_AW_norm_df(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}};
    f_name::Union{Nothing, String}=nothing,
    psize::Union{Nothing, Tuple}=nothing,
    logscale::Bool=false
)
    # Define labels
    label_A = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    label_W = latexstring(logscale ? "\\log(||\\tilde{W} - W||_F)" : "||\\tilde{W} - W||_F")

    # Convert norms into DataFrames
    df_A = norms_to_dataframe_A(data; logscale=logscale)
    df_W = norms_to_dataframe_W(data; logscale=logscale)

    plots = []

    # --- Plot A ---
    p_A = @df df_A plot(
        :Time,
        :Norm,
        group=:Label,
        ribbon=(x -> std(x)).(groupby(df_A, [:Label, :Time])), # Adds ribbons per group if needed
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        ylabel=label_A,
        legend=:topleft,
        xlabel=false,
        palette=:tab10
    )
    push!(plots, p_A)

    # --- Plot W ---
    p_W = @df df_W plot(
        :Time,
        :Norm,
        group=:Label,
        ribbon=(x -> std(x)).(groupby(df_W, [:Label, :Time])), # Adds ribbons per group if needed
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        xlabel="time [s]",
        ylabel=label_W,
        legend=:topleft,
        palette=:tab10
    )
    push!(plots, p_W)

    # --- Combine plots ---
    plt = plot(
        plots...,
        layout=(2,1),
        link=:x
    )

    # --- Style ---
    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    # --- Save if filename given ---
    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end

function plot_param_AW_norm_old(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, logscale::Bool=false)
    label_A = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    label_W = latexstring(logscale ? "\\log(||\\tilde{W} - W||_F)" : "||\\tilde{W} - W||_F")
    plots = []

    #p_A = plot(xlabel="time [s]", ylabel=label_A)
    p_A = plot(ylabel=label_A)

    for (label, vrecs, vsys) in data
        norms = compute_A_norms(vrecs, vsys; logscale=logscale)
        mean_norm = vec(mean(norms, dims=1))
        std_norm = vec(std(norms, dims=1))
        plot!(mean_norm, ribbon=std_norm, lw=DEFAULTS.linewidth, label=label, fillalpha=DEFAULTS.fillalpha)
    end
    push!(plots, p_A)

    p_W = plot(xlabel="time [s]", ylabel=label_W)
    for (label, vrecs, vsys) in data[1:2]
        norms = compute_W_norms(vrecs, vsys; logscale=logscale)
        mean_norm = vec(mean(norms, dims=1))
        std_norm = vec(std(norms, dims=1))
        plot!(mean_norm, ribbon=std_norm, lw=DEFAULTS.linewidth, label=label, fillalpha=DEFAULTS.fillalpha)
    end
    push!(plots, p_W)

    p = plot(plots..., layout=(2,1))

    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=8Plots.mm,
        left_margin=12Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end
    return p
end

function plot_param_W_norm(recs::Matrix{Recorder}, syss::Matrix{MARXSystem}, T::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    N_runs, n_Ts = size(recs)
    (D_y, D_x, N) = size(recs[1,T].Ws)

    p = plot(xlabel="time [s]")
    palette = theme_palette(:auto)
    color = palette[1]
    norms = zeros(N_runs, N)
    for i in 1:N_runs
        W_diff = syss[i, T].W .- recs[i, T].Ws
        norms[i,:] = [ norm(W_diff[:,:,t],2) for t in 1:N ]
    end
    plot!(mean(norms, dims=1)', ribbon=std(norms, dims=1), label=latexstring("||W||_F"), color=color, lw=DEFAULTS.linewidth)
    hline!([0.0], label=nothing, lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_W_norm(rec::Recorder; sys::Union{Nothing, System}=nothing, f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_y, D_y, N) = size(rec.Ws)
    palette = theme_palette(:auto)
    color = palette[1]
    p = plot([ norm(rec.Ws[:,:,t], 2) for t in 1:N ], label=latexstring("||W||_F"), color=color, linewidth=DEFAULTS.linewidth, xlabel="time [s]")
    if !isnothing(sys)
        hline!([norm(sys.W, 2)], lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle, label=latexstring("||\\tilde{W}||_F"), color=color)
    end
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=12Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_pdf_params(rec::Recorder, sys::System; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    p = plot(xlabel="time [s]", ylabel=label_pdf_params_true)
    (D_y, D_x, N) = size(rec.Ms)
    pdf_AW = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t]) for t in 1:N ]
    plot!(pdf_AW, lw=DEFAULTS.linewidth, label=nothing)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_pdf_params(recs::Matrix{Recorder}, syss::Matrix{MARXSystem}, T::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    N_runs, n_Ts = size(recs)
    (D_y, D_x, N) = size(recs[1,T].Ms)

    pdfs_AW = zeros(N_runs, N)
    for i in 1:N_runs
        rec = recs[i,T]
        sys = syss[i,T]
        pdfs_AW[i,:] = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t]) for t in 1:N ]
    end

    p = plot(xlabel="time [s]", ylabel=label_pdf_params_true)
    plot!(mean(pdfs_AW, dims=1)', ribbon=std(pdfs_AW, dims=1), lw=DEFAULTS.linewidth, label=nothing)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plots_paper_single(
    data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}};
    f_name::Union{Nothing, String}=nothing,
    psize::Union{Nothing, Tuple}=nothing,
    logscale::Bool=false,
    run_index::Union{Nothing,Int}=nothing,
    a_indices::Union{Nothing, Vector{Tuple{Int64, Int64}}}=nothing
)
    # Define labels
    label_A = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    label_W = latexstring(logscale ? "\\log(||\\tilde{W} - W||_F)" : "||\\tilde{W} - W||_F")
    #label_pdf = latexstring("-\\log p(\\tilde{\\Theta} \\mid \\mathcal{D}_k)")
    label_pdf = latexstring("-\\log p(\\tilde{A},\\tilde{W} \\mid \\mathcal{D}_k)")

    # Convert norms into DataFrames
    df_A = norms_to_dataframe_A(data; logscale=logscale)
    df_W = norms_to_dataframe_W(data; logscale=logscale)

    # --- Compute mean and std ---
    grouped_A = groupby(df_A, [:Label, :Time])
    meanstd_A = combine(grouped_A,
        :Norm => mean => :MeanNorm,
        :Norm => std => :StdNorm
    )

    grouped_W = groupby(df_W, [:Label, :Time])
    meanstd_W = combine(grouped_W,
        :Norm => mean => :MeanNorm,
        :Norm => std => :StdNorm
    )

    # --- Create PDF DataFrame ---
    pdf_records = DataFrame(Label=String[], Time=Int[], PDF=Float64[])
    for (data_label, vrecs, vsys) in data
        if data_label == "RLS" continue end
        N_runs = length(vrecs)
        (D_y, D_x, N) = size(vrecs[1].Ms)
        for i in 1:N_runs
            rec = vrecs[i]
            sys = vsys[i]
            pdf_vals = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t], D_x, D_y) for t in 1:N]
            for (t_idx, pdf_val) in enumerate(pdf_vals)
                push!(pdf_records, (data_label, t_idx, pdf_val))
            end
        end
    end

    # Group and compute mean/std for PDF
    grouped_pdf = groupby(pdf_records, [:Label, :Time])
    meanstd_pdf = combine(grouped_pdf,
        :PDF => mean => :MeanPDF,
        :PDF => std => :StdPDF
    )

    # --- Plot A ---
    p_A = @df meanstd_A plot(
        :Time,
        :MeanNorm,
        ribbon=:StdNorm,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        ylabel=label_A,
        #legend=:topright,
        legend=:top,
        legend_columns=3, # <<== Wide legend here
        legend_border=false,

        palette=:tab10
    )

    # --- Plot W ---
    p_W = @df meanstd_W plot(
        :Time,
        :MeanNorm,
        ribbon=:StdNorm,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        #xlabel="time [s]",
        ylabel=label_W,
        legend=nothing,
        palette=:tab10
    )

    # --- Plot PDF ---
    p_pdf = @df meanstd_pdf plot(
        :Time,
        :MeanPDF,
        ribbon=:StdPDF,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        xlabel="time [s]",
        ylabel=label_pdf,
        legend=nothing,
        palette=:tab10
    )

    #plots_to_stack = [p_A, p_W, p_pdf]
    plots_to_stack = [p_A, p_W]

        # --- Third plot: single recorder ---
    if !isnothing(run_index)
        (label_single, vrecs, vsys) = data[1]  # assuming only one data set
        rec = vrecs[run_index]
        sys = vsys[run_index]

        (D_y, D_x, N) = size(rec.Ms)

        # Compute std_A
        Σs = zeros(D_y*D_x, D_y*D_x, N)
        std_A = zeros(D_y, D_x, N)
        for t in 1:N
            Σs[:,:,t] = kron(inv(rec.Λs[:,:,t]), inv(rec.Ws[:,:,t]))
            std_A[:,:,t] = reshape(sqrt.(diag(Σs[:,:,t])), (D_y, D_x))
        end

        # Prepare new color palette (different from others)
        palette_A = palette(:Set1)
        #palette_A = palette(:Paired)
        palette_W = palette(:Pastel1)
        #palette_W = palette(:Accent)
        palette_W = palette(:Dark2_8)
        subscripts_digit = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

        # Create empty plot
        p_single = plot(
            #xlabel="time [s]",
            ylabel=latexstring("p(\\tilde{A} \\mid \\mathcal{D}_t)"),
            legend=:bottom,
            legend_border=false,
            legend_columns=length(a_indices),
            palette=palette_A
        )

        cidx = 1
        selected_indices = isnothing(a_indices) ? collect(Iterators.product(1:D_y, 1:D_x)) : a_indices

        for (i, j) in selected_indices
            color = palette_A[(cidx - 1) % length(palette_A) + 1]
            label_appendix = (
                i <= length(subscripts_digit) && j <= length(subscripts_digit) ?
                "$(subscripts_digit[i]),$(subscripts_digit[j])" : "$i,$j"
            )
            label_text = "ã$label_appendix"

            # Mean prediction with ribbon
            plot!(
                1:N,
                rec.Ms[i,j,:],
                ribbon=std_A[i,j,:],
                #label=label_text,
                label=nothing,
                color=color,
                lw=DEFAULTS.linewidth,
                fillalpha=DEFAULTS.fillalpha
            )

            # True A value (dashed line)
            plot!(
                1:N,
                rec.As[i,j,:],
                #label=nothing,
                label=label_text,
                color=color,
                lw=DEFAULTS.linewidth,
                ls=DEFAULTS.linestyle
            )

            cidx += 1
        end

        push!(plots_to_stack, p_single)

                # --- Fourth plot: W estimation ---
        (D_y, _, N) = size(rec.Ωs)

        # Compute W and std_W
        Ws = zeros(D_y, D_y, N)
        std_W = zeros(D_y, D_y, N)
        for t in 1:N
            iΩ = inv(rec.Ωs[:,:,t])
            Ws[:,:,t] = rec.νs[t] * iΩ
            for i in 1:D_y
                for j in 1:D_y
                    std_W[i,j,t] = sqrt(rec.νs[t]*(iΩ[i,j]^2 + iΩ[i,i]*iΩ[j,j]))
                end
            end
        end

        # Create W plot
        p_single_W = plot(
            xlabel="time [s]",
            ylabel=latexstring("p(\\tilde{W} \\mid \\mathcal{D}_t)"),
            legend=:top,
            legend_columns=3,
            legend_border=false,
            palette=palette_W
        )

        cidx = 1
        for i in 1:D_y
            for j in 1:D_y
                if i == 2 && j == 1 continue end
                color = palette_W[((i-1)*D_y + j - 1) % length(palette_W) + 1]
                label_appendix = (
                    i <= length(subscripts_digit) && j <= length(subscripts_digit) ?
                    "$(subscripts_digit[i]),$(subscripts_digit[j])" : "$i,$j"
                )
                label_text = "w̃$label_appendix"

                # Mean prediction with ribbon
                plot!(
                    1:N,
                    Ws[i,j,:],
                    ribbon=std_W[i,j,:],
                    #label=label_text,
                    label=nothing,
                    color=color,
                    lw=DEFAULTS.linewidth,
                    fillalpha=DEFAULTS.fillalpha
                )

                # True W value (constant line)
                if !isnothing(sys)
                    hline!(
                        [sys.W[i,j]],
                        #label=nothing,
                        label=label_text,
                        color=color,
                        lw=DEFAULTS.linewidth,
                        ls=DEFAULTS.linestyle
                    )
                end

                cidx += 1
            end
        end

        push!(plots_to_stack, p_single_W)

    end


    # --- Combine plots ---
    plt = plot(
        plots_to_stack...,
        layout=(length(plots_to_stack),1),
        link=:x
    )

    # --- Style ---
    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=6Plots.mm,
        left_margin=8Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end

function plots_paper(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}};
    f_name::Union{Nothing, String}=nothing,
    psize::Union{Nothing, Tuple}=nothing,
    logscale::Bool=false
)
    # Define labels
    label_A = latexstring(logscale ? "\\log(||\\tilde{A} - A||_F)" : "||\\tilde{A} - A||_F")
    label_W = latexstring(logscale ? "\\log(||\\tilde{W} - W||_F)" : "||\\tilde{W} - W||_F")
    #label_pdf = latexstring("-\\log p(\\tilde{\\Theta} \\mid \\mathcal{D}_k)")
    label_pdf = latexstring("-\\log p(\\tilde{A},\\tilde{W} \\mid \\mathcal{D}_k)")

    # Convert norms into DataFrames
    df_A = norms_to_dataframe_A(data; logscale=logscale)
    df_W = norms_to_dataframe_W(data; logscale=logscale)

    # --- Compute mean and std ---
    grouped_A = groupby(df_A, [:Label, :Time])
    meanstd_A = combine(grouped_A,
        :Norm => mean => :MeanNorm,
        :Norm => std => :StdNorm
    )

    grouped_W = groupby(df_W, [:Label, :Time])
    meanstd_W = combine(grouped_W,
        :Norm => mean => :MeanNorm,
        :Norm => std => :StdNorm
    )

    # --- Create PDF DataFrame ---
    pdf_records = DataFrame(Label=String[], Time=Int[], PDF=Float64[])
    for (data_label, vrecs, vsys) in data
        if data_label == "RLS" continue end
        N_runs = length(vrecs)
        (D_y, D_x, N) = size(vrecs[1].Ms)
        for i in 1:N_runs
            rec = vrecs[i]
            sys = vsys[i]
            pdf_vals = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t], D_x, D_y) for t in 1:N]
            for (t_idx, pdf_val) in enumerate(pdf_vals)
                push!(pdf_records, (data_label, t_idx, pdf_val))
            end
        end
    end

    # Group and compute mean/std for PDF
    grouped_pdf = groupby(pdf_records, [:Label, :Time])
    meanstd_pdf = combine(grouped_pdf,
        :PDF => mean => :MeanPDF,
        :PDF => std => :StdPDF
    )

    # --- Plot A ---
    p_A = @df meanstd_A plot(
        :Time,
        :MeanNorm,
        ribbon=:StdNorm,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        ylabel=label_A,
        #legend=:topright,
        legend=:top,
        legend_columns=3, # <<== Wide legend here
        legend_border=false,

        palette=:tab10
    )

    # --- Plot W ---
    p_W = @df meanstd_W plot(
        :Time,
        :MeanNorm,
        ribbon=:StdNorm,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        #xlabel="time [s]",
        ylabel=label_W,
        legend=nothing,
        palette=:tab10
    )

    # --- Plot PDF ---
    p_pdf = @df meanstd_pdf plot(
        :Time,
        :MeanPDF,
        ribbon=:StdPDF,
        group=:Label,
        lw=DEFAULTS.linewidth,
        fillalpha=DEFAULTS.fillalpha,
        xlabel="time [s]",
        ylabel=label_pdf,
        legend=nothing,
        palette=:tab10
    )

    # --- Combine plots ---
    plt = plot(
        p_A, p_W, p_pdf,
        layout=(3,1),
        link=:x
    )

    # --- Style ---
    plot!(
        size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi, DEFAULTS.height_in*DEFAULTS.dpi) : psize,
        tickfontsize = DEFAULTS.tickfontsize,
        titlefontsize = DEFAULTS.titlefontsize,
        legendfontsize = DEFAULTS.legendfontsize,
        guidefontsize = DEFAULTS.guidefontsize,
        bottom_margin=6Plots.mm,
        left_margin=8Plots.mm
    )

    if !isnothing(f_name)
        savefig(f_name)
    end

    return plt
end


function plots_paper_x(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, xlabel::String="Training size T")
    #p = plot(xlabel="time [s]", ylabel=latexstring("-\\log p(\\tilde{\\Theta} \\mid \\mathcal{D}_k)"))
    p = plot(xlabel="time [s]", ylabel=latexstring("-\\log p(\\tilde{A}, \\tilde{W} \\mid \\mathcal{D}_k)"))
    for (data_label, vrecs, vsys) in data
        N_runs = size(vrecs)[1]
        (D_y, D_x, N) = size(vrecs[1].Ms)
        pdfs_AW = zeros(N_runs, N)
        for i in 1:N_runs
            rec = vrecs[i]
            sys = vsys[i]
            pdfs_AW[i,:] = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t], D_x, D_y) for t in 1:N ]
        end
        plot!(mean(pdfs_AW, dims=1)', ribbon=std(pdfs_AW, dims=1)', lw=DEFAULTS.linewidth, label=data_label)
    end
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_pdf_params(data::Vector{Tuple{String, Vector{Recorder}, Vector{MARXSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, xlabel::String="Training size T")
    p = plot(xlabel="time [s]", ylabel=label_pdf_params_true)
    for (data_label, vrecs, vsys) in data
        N_runs = size(vrecs)[1]
        (D_y, D_x, N) = size(vrecs[1].Ms)
        pdfs_AW = zeros(N_runs, N)
        for i in 1:N_runs
            rec = vrecs[i]
            sys = vsys[i]
            pdfs_AW[i,:] = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t], D_x, D_y) for t in 1:N ]
        end
        plot!(mean(pdfs_AW, dims=1)', ribbon=std(pdfs_AW, dims=1)', lw=DEFAULTS.linewidth, label=data_label)
    end
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_pdf_params(data::Vector{Tuple{String, Vector{Recorder}, Vector{DoubleMassSpringDamperSystem}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, xlabel::String="Training size T")
    p = plot(xlabel="time [s]", ylabel=label_pdf_params_true)
    for (data_label, vrecs, vsys) in data
        N_runs = size(vrecs)[1]
        (D_y, D_x, N) = size(vrecs[1].Ms)
        pdfs_AW = zeros(N_runs, N)
        for i in 1:N_runs
            rec = vrecs[i]
            sys = vsys[i]
            pdfs_AW[i,:] = [pdf_params(sys.A, sys.W, rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t], D_x, D_y) for t in 1:N ]
        end
        plot!(mean(pdfs_AW, dims=1)', ribbon=std(pdfs_AW, dims=1)', lw=DEFAULTS.linewidth, label=data_label)
    end
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(bottom_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_pdf_predictive(data::Vector{Tuple{String, Vector{Recorder}, Vector{S}}}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, xlabel::String="Training size T") where {S<:System}
    p = plot(xlabel="time [s]", ylabel=label_surprisals)
    for (data_label, vrecs, vsys) in data
        N_runs = size(vrecs)[1]
        (D_y, D_x, N) = size(vrecs[1].Ms)
        pdfs = zeros(N_runs, N)
        for i in 1:N_runs
            rec = vrecs[i]
            sys = vsys[i]
            # pdf_predictive(y, x, M, Λ, Ω, ν, D_y, logpdf=true)
            pdfs[i,:] = [pdf_predictive(rec.ys[:,t], rec.xs[:,t], rec.Ms[:,:,t], rec.Λs[:,:,t], rec.Ωs[:,:,t], rec.νs[t], D_y, logpdf=true) for t in 1:N ]
        end
        plot!(mean(pdfs, dims=1)', ribbon=std(pdfs, dims=1), lw=DEFAULTS.linewidth, label=data_label, fillalpha=0.3)
    end
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_surprisals(rec::Recorder; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    p = plot(xlabel="time [s]", ylabel=latexstring("-\\log p(\\tilde{y}_t \\mid \\mathcal{D}_t)"))
    plot!(rec.surprisals, label=nothing, lw=DEFAULTS.linewidth)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_ces_posterior_likelihood(rec::Recorder; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    p = plot(xlabel="time [s]", ylabel=latexstring("H[q(\\Theta \\mid \\mathcal{D}_t), p(\\tilde{y}_t \\mid \\Theta, x_t)]"))
    plot!(rec.ces_posterior_likelihood, label=nothing, lw=DEFAULTS.linewidth)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_ces_posterior_prior(rec::Recorder; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    p = plot(xlabel="time [s]", ylabel=latexstring("H[q(\\Theta \\mid \\mathcal{D}_t), p(\\Theta \\mid \\mathcal{D}_{t-1})]"))
    plot!(rec.ces_posterior_prior, label=nothing, lw=DEFAULTS.linewidth)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_es_posterior(rec::Recorder; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    N = size(rec.es_posterior)[1] # TODO: Fix off-by-one
    p = plot(xlabel="time [s]", ylabel=label_es_posterior)
    plot!(rec.es_posterior[1:N-1], lw=DEFAULTS.linewidth, label=nothing)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_ces_posterior_likelihood_kls_posterior_prior(rec::Recorder; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, show_surprisals::Bool=false)
    N = size(rec.surprisals)[1] # TODO: fix off-by-one error N-1
    p = plot(xlabel="time [s]")
    if show_surprisals
        plot!(rec.surprisals[1:N-1], lw=DEFAULTS.linewidth, label="surprisal", ls=:dash)
    end
    plot!(rec.ces_posterior_likelihood[1:N-1], lw=DEFAULTS.linewidth, label="accuracy")
    plot!(rec.kls_posterior_prior[1:N-1], lw=DEFAULTS.linewidth, label="complexity")
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_ces_posterior_likelihood_ces_posterior_prior_es_posterior(rec::Recorder; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, show_surprisals::Bool=false)
    p = plot(xlabel="time [s]")
    if show_surprisals
        plot!(rec.surprisals, lw=DEFAULTS.linewidth, ls=:dash, label=label_surprisals)
    end
    plot!(rec.ces_posterior_likelihood, lw=DEFAULTS.linewidth, label=label_ces_posterior_likelihood)
    plot!(rec.ces_posterior_prior, lw=DEFAULTS.linewidth, label=label_ces_posterior_prior)
    plot!(rec.es_posterior, lw=DEFAULTS.linewidth, label=label_es_posterior)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_Ω(agent::MARXAgent, sys::System; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_y, D_y) = size(agent.Ω)
    xticks = yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        xticks = yticks = (1:D_y, ["yₜ,₁", "yₜ,₂"])
    end
    p = heatmap(agent.Ω, xticks=xticks, yticks=yticks)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    #plot!(bottom_margin=4Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_Ω(rec::Recorder, sys::System, t::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_y, D_y, N) = size(rec.Ωs)
    xticks = yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        xticks = yticks = (1:D_y, ["yₜ,₁", "yₜ,₂"])
    end
    p = heatmap(rec.Ωs[:,:,t], xticks=xticks, yticks=yticks, yflip=true)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    #plot!(bottom_margin=4Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_Ω_timeseries(rec::Recorder)
    p = plot()
    (D_y, D_y, N) = size(rec.Ωs)
    for dim in 1:D_y
        plot!(rec.Ωs[dim, :, :]')
    end
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_W_timeseries(rec::Recorder; sys::Union{Nothing, System}=nothing, f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    subscripts_digit = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    p = plot(xlabel="time [s]", ylabel=label_pdf_param_W)
    (D_y, D_y, N) = size(rec.Ωs)
    Ws = zeros(D_y, D_y, N)
    std_W = zeros(D_y, D_y, N)
    for t in 1:N
        iΩ = inv(rec.Ωs[:,:,t])
        Ws[:,:,t] = rec.νs[t] * iΩ
        for i in 1:D_y
            for j in i:D_y
                std_W[i,j,t] = sqrt(rec.νs[t]*(iΩ[i,j]^2 + iΩ[i,i]*iΩ[j,j]))
            end
        end
    end
    palette = theme_palette(:auto)
    for i in 1:D_y
        for j in i:D_y
            color = palette[((i-1)*D_y + j - 1) % length(palette) + 1]
            label_appendix = "$(subscripts_digit[i]),$(subscripts_digit[j])"
            plot!(Ws[i, j, :], ribbon=std_W[i, j, :], label=nothing, fillalpha=DEFAULTS.fillalpha, color=color, lw=DEFAULTS.linewidth)
            if !isnothing(sys)
                hline!([sys.W[i, j]], label="w̃$label_appendix", color=color, lw=DEFAULTS.linewidth, ls=DEFAULTS.linestyle)
            end
        end
    end
    #plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    #plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    #plot!(bottom_margin=8Plots.mm)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_ΛΩ(rec::Recorder, agent::Agent, sys::System, t::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_x, D_x, N) = size(rec.Λs)
    (D_y, D_y, N) = size(rec.Ωs)
    plots = []
    xticks = yticks = (1:D_x, get_memory_ticks(agent.memory_type, D_y, agent.N_y, agent.N_u))
    push!(plots, heatmap(rec.Λs[:,:,t], xticks=xticks, yticks=yticks, yflip=true, xrotation=-45))

    xticks = yticks = (1:D_y, latexstring("y_{$(i)}") for i in 1:D_y) # [ latexstring("y_{$(i)}") for i in 1:D_y ]
    if typeof(sys) == MARXSystem
        xticks = yticks = (1:D_y, ["yₜ,₁", "yₜ,₂"])
    end
    push!(plots, heatmap(rec.Ωs[:,:,t], xticks=xticks, yticks=yticks, yflip=true))
    n_plots = length(plots)
    p = plot(plots..., layout=(1, n_plots))
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(bottom_margin=12Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_Λ(rec::Recorder, sys::System, agent::Agent, t::Int; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_x, D_x, N) = size(rec.Λs)
    ticks = get_memory_ticks(agent.memory_type, agent.D_y, agent.N_y, agent.N_u)
    xticks = yticks = (1:D_x, ticks)
    p = heatmap(rec.Λs[:,:,t], xticks=xticks, yticks=yticks, yflip=true, xrotation=-45)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(bottom_margin=12Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_param_Λ(rec::Recorder, agent::Agent, dim::Int)
    (D_x, D_x, N) = size(rec.Λs)
    plots = []
    for i in 1:D_x
        time_series = rec.Λs[i, dim:dim, :]  # Extract the 1000 values for the i-th dimension
        p = plot(time_series[:], label=nothing)
        push!(plots, p)
    end
    p = plot(plots..., layout=(D_x, 1))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_Λ_timeseries(rec::Recorder)
    p = plot()
    (D_x, D_x, N) = size(rec.Λs)
    for dim in 1:D_x
        plot!(rec.Λs[dim, :, :]', label=nothing)
    end
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_Λ_timeseries_dim(rec::Recorder, dim::Int)
    p = plot()
    (D_x, D_x, N) = size(rec.Λs)
    p = plot(rec.Λs[dim, :, :]')
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_M_combos(rec::Recorder, agent::Agent, dim_y::Int, dim_x::Int, t_beg::Int, t_step::Int)
    # Store dimensions in a list for easier referencing
    dimensions = [vec(rec.Ms[dim, i, t_beg:t_beg+t_step]) for i in 1:agent.D_x]

    # Define the pairs of dimensions to plot
    dimension_pairs = combinations_without_repetition(agent.D_y, agent.D_x)

    # Create the plot
    p = plot(layout = (agent.D_x, agent.D_y), xlabel = "X", ylabel = "Y", legend = false)

    # Loop over each pair and create a subplot for it
    for (i, (dim_x, dim_y)) in enumerate(dimension_pairs)
        plot!(dimensions[dim_x], dimensions[dim_y], subplot = i, xlabel = "Dim $dim_x", ylabel = "Dim $dim_y", title = "($dim_x, $dim_y)")
    end
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_M_combos(rec::Recorder, agent::Agent, dim::Int, t_beg::Int, t_step::Int)
    # Store dimensions in a list for easier referencing
    dimensions = [vec(rec.Ms[dim, i, t_beg:t_beg+t_step]) for i in 1:agent.D_x]

    # Define the pairs of dimensions to plot
    dimension_pairs = combinations_without_repetition(agent.D_y, agent.D_x)

    # Create the plot
    p = plot(layout = (agent.D_x, agent.D_y), xlabel = "X", ylabel = "Y", legend = false)

    # Loop over each pair and create a subplot for it
    for (i, (dim_x, dim_y)) in enumerate(dimension_pairs)
        plot!(dimensions[dim_x], dimensions[dim_y], subplot = i, xlabel = "Dim $dim_x", ylabel = "Dim $dim_y", title = "($dim_x, $dim_y)")
    end
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_Ω_combos(rec::Recorder, agent::Agent, dim::Int, t_beg::Int, t_step::Int)
    # Store dimensions in a list for easier referencing
    dimensions = [vec(rec.Ωs[i, dim, t_beg:t_beg+t_step]) for i in 1:agent.D_y]

    # Define the pairs of dimensions to plot
    dimension_pairs = [(1, 2)]

    # Create the plot
    p = plot(layout = (1,1), xlabel = "X", ylabel = "Y", legend = false)

    # Loop over each pair and create a subplot for it
    for (i, (dim_x, dim_y)) in enumerate(dimension_pairs)
        plot!(dimensions[dim_x], dimensions[dim_y], subplot = i, xlabel = "Dim $dim_x", ylabel = "Dim $dim_y", title = "($dim_x, $dim_y)")
    end
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_param_Λ_combos(rec::Recorder, agent::Agent, dim::Int, t_beg::Int, t_step::Int)
    # Store dimensions in a list for easier referencing
    dimensions = [vec(rec.Us[i, dim, t_beg:t_beg+t_step]) for i in 1:agent.D_x]

    # Define the pairs of dimensions to plot
    dimension_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

    # Create the plot
    p = plot(layout = (2,3), xlabel = "X", ylabel = "Y", legend = false)

    # Loop over each pair and create a subplot for it
    for (i, (dim_x, dim_y)) in enumerate(dimension_pairs)
        plot!(dimensions[dim_x], dimensions[dim_y], subplot = i, xlabel = "Dim $dim_x", ylabel = "Dim $dim_y", title = "($dim_x, $dim_y)")
    end
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

# agent.Λ = Λ0 + x*x'
function plot_update_Λ(rec::Recorder, t_beg::Int)
    plots = []
    x = rec.xs[:,t_beg]
    X = x*x'
    Λ = rec.Λs[:,:,t_beg-1]
    (D_x, D_x) = size(Λ)
    Λ_new = Λ + X
    Λ_diff = Λ_new - rec.Λs[:,:,t_beg]
    @assert Λ_diff == zeros(D_x, D_x) "Invalid Λ update code"
    push!(plots, heatmap(X, title="X"))
    push!(plots, heatmap(Λ, title="Λ"))
    push!(plots, heatmap(X + Λ, title="Λ+X"))
    push!(plots, heatmap(inv(X + Λ), title="inv(Λ+X)"))
    p = plot(plots..., layout=(2,2))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

# agent.Ω = Ω0 + y*y' + M0*Λ0*M0' - agent.M*inv(agent.Λ)*agent.M'
function plot_update_Ω(rec::Recorder, t_beg::Int)
    plots = []
    y = rec.ys[:,t_beg]
    Ω0 = rec.Ωs[:,:,t_beg-1]
    (D_y, D_y) = size(Ω0)
    Y = y*y'
    M0 = rec.Ms[:,:,t_beg-1]
    Λ0 = rec.Λs[:,:,t_beg-1]
    M = rec.Ms[:,:,t_beg]
    Λ = rec.Λs[:,:,t_beg]
    M0ΛM0 = M0*Λ0*M0'
    MΛM = M*Λ*M'
    Ω = Ω0 + Y + M0ΛM0 - MΛM
    Ω_diff = Ω - rec.Ωs[:,:,t_beg]
    @assert Ω_diff == zeros(D_y, D_y) "Invalid Ω update code"
    push!(plots, heatmap(Ω, title="Λ"))
    push!(plots, heatmap(Y, title="Y"))
    push!(plots, heatmap(M0ΛM0, title="M0ΛM0"))
    push!(plots, heatmap(-MΛM, title="-MΛM"))
    push!(plots, heatmap(Ω + Y, title="Ω + Y"))
    push!(plots, heatmap(Ω + Y + M0ΛM0, title="Ω + Y + M0ΛM0"))
    push!(plots, heatmap(Ω + Y + M0ΛM0 - MΛM, title="Ω + Y + M0ΛM0 - MΛM"))
    push!(plots, heatmap(inv(Ω), title="inv(Ω)"))
    p = plot(plots..., layout=(2,4))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

    # M: D_x × D_y
    #agent.M = (M0*Λ0 + y*x')*inv(agent.Λ)
function plot_update_M(rec::Recorder, t_beg::Int)
    plots = []
    x = rec.xs[:,t_beg]
    y = rec.ys[:,t_beg]
    YX = y*x'
    Λ = rec.Λs[:,:,t_beg-1]
    M = rec.Ms[:,:,t_beg-1]
    MU = M*Λ
    yxMU = YX + MU
    Λ_ = rec.Λs[:,:,t_beg]
    M_ = yxMU*inv(Λ_)
    push!(plots, heatmap(M, title="M"))
    push!(plots, heatmap(Λ, title="Λ"))
    push!(plots, heatmap(MU, title="MΛ"))
    push!(plots, heatmap(YX, title="yx'"))
    push!(plots, heatmap(yxMU, title="MΛ + yx'"))
    push!(plots, heatmap(Λ_, title="Λ_"))
    push!(plots, heatmap(M_, title="M_ = (MΛ + yx')*inv(Λ_)"))
    p = plot(plots..., layout=(2,4))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_det_ΛΩ(rec::Recorder)
    (_, _, N) = size(rec.Λs)
    det_Λ = [ det(rec.Λs[:,:,t]) for t in 1:N ]
    det_Ω = [ det(rec.Ωs[:,:,t]) for t in 1:N ]
    det_Σ = [ det(rec.pred_Σs[:,:,t]) for t in 1:N ]
    plots = []
    push!(plots, plot(det_Σ, label="det(Ψ^{-1})"))
    push!(plots, plot(det_Λ, label="det(Λ)"))
    push!(plots, plot(det_Ω, label="det(Ω)"))
    n_plots = length(plots)
    p = plot(plots..., layout=(n_plots, 1))
    plot!(size=(DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    return p
end

function plot_rmse_boxplot(rmses::Array{Float64, 3}, Ts::Vector{Int}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_y, N_runs, n_Ts) = size(rmses)
    xticks = (1:n_Ts, Ts)
    plots = []
    xlabel = "Training size T"
    ylabels = ["y₁", "y₂"]
    for dim in 1:D_y
        push!(plots, boxplot(rmses[dim,:,:], ylabel="rmse $(ylabels[dim])", boxstyle=:auto, xticks=dim == 1 ? nothing : xticks, label=nothing))
        if dim == D_y xlabel!(xlabel) end
    end
    n_plots = length(plots)
    p = plot(plots..., layout=(2,1))
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    plot!(bottom_margin=12Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_rmse_line(rmses::Array{Float64, 3}, Ts::Vector{Int}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    (D_y, N_runs, n_Ts) = size(rmses)
    xticks = (1:n_Ts, Ts)
    plots = []
    xlabel = "Training size T"
    ylabels = ["y₁", "y₂"]
    for dim in 1:D_y
        μs = [mean(rmses[dim,:,T]) for T in 1:n_Ts]
        σs = [std(rmses[dim,:,T]) for T in 1:n_Ts]
        push!(plots, plot(Ts, μs, ribbon=σs, ylabel="rmse $(ylabels[dim])", label=ylabels[dim]))
        if dim == D_y xlabel!(xlabel) end
    end
    n_plots = length(plots)
    p = plot(plots..., layout=(2,1))
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    plot!(right_margin=8Plots.mm)
    plot!(left_margin=8Plots.mm)
    plot!(bottom_margin=12Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_rmse_line_baseline(rmses::Vector{Tuple{String, Array{Float64, 3}}}, Ts::Vector{Int}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing, xlabel::String="Training size T", logscale::Bool=false)
    p = plot()
    for (rmses_label, rmses_data) in rmses
        (D_y, N_runs, n_Ts) = size(rmses_data)
        xticks = (1:n_Ts, Ts)
        ylabel = "RMSE"
        for dim in 1:D_y
            if logscale
                μs = [mean(log.(rmses_data[dim,:,T])) for T in 1:n_Ts]
                σs = [std(log.(rmses_data[dim,:,T])) for T in 1:n_Ts] ./ sqrt(N_runs)
                println(μs)
            else
                μs = [mean(rmses_data[dim,:,T]) for T in 1:n_Ts]
                σs = [std(rmses_data[dim,:,T]) for T in 1:n_Ts] ./ sqrt(N_runs)
            end
            plot!(Ts, μs, ribbon=σs, ylabel=ylabel, label=rmses_label, linewidth=DEFAULTS.linewidth, fillalpha=DEFAULTS.fillalpha)
            if dim == D_y xlabel!(xlabel) end
        end
    end
    #plot!(legendfontsize=8)
    plot!(grid=false)
    plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    plot!(tickfontsize = DEFAULTS.tickfontsize, titlefontsize = DEFAULTS.titlefontsize, legendfontsize = DEFAULTS.legendfontsize, guidefontsize = DEFAULTS.guidefontsize)
    plot!(bottom_margin=8Plots.mm, left_margin=8Plots.mm)
    if !isnothing(f_name) savefig(f_name) end
    return p
end

function plot_rmse_violin(rmses::Vector{Tuple{String, Array{Float64, 3}}}, Ts::Vector{Int}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple{Int,Int}}=nothing, xlabel::String="Training size T")
    df = DataFrame(T = Int[], RMSE = Float64[], Estimator = String[])

    for (rmses_label, rmses_data) in rmses
        (D_y, N_runs, n_Ts) = size(rmses_data)

        for t_index in 1:n_Ts
            for run in 1:N_runs
                rmse_value = rmses_data[1, run, t_index]
                push!(df, (T = Ts[t_index], RMSE = rmse_value, Estimator = rmses_label))
            end
        end
    end

    #df.T = string.(df.T)
    p = violin(df.T, df.RMSE, group = df.Estimator, xlabel = xlabel, ylabel = "RMSE", legend = :topleft, title = "RMSE distributions for different estimators")

    if !isnothing(psize) p.size = psize end
    if !isnothing(f_name) savefig(p, f_name) end

    return p
end

function plot_rmse_heatmap_Nys_Nus(rmses::Array{Float64, 4}, y_dim::Int, Nys::Vector{Int}, Nus::Vector{Int}; f_name::Union{Nothing, String}=nothing, psize::Union{Nothing, Tuple}=nothing)
    rmses_mean = dropdims(mean(rmses[y_dim,:,:,:], dims=1), dims=1)
    rmses_std = dropdims(std(rmses[y_dim,:,:,:], dims=1), dims=1)
    n_Nys, n_Nus = size(rmses_mean)
    p = heatmap(rmses_mean, color=:viridis, fillalpha=0.8, xlabel="N_u", ylabel="N_y", xticks=(1:length(Nus), Nus), yticks=(1:length(Nys), Nys))
    for i in 1:n_Nys, j in 1:n_Nus
        annotation_text = "$(round(rmses_mean[i, j], digits=3)) ± $(round(rmses_std[i, j], digits=2))"
        annotate!(j, i, text(annotation_text, :black, 8, :center))
    end
    #plot!(size = isnothing(psize) ? (DEFAULTS.width_in*DEFAULTS.dpi,DEFAULTS.height_in*DEFAULTS.dpi) : psize)
    #plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    if !isnothing(f_name) savefig(f_name) end
     #x = repeat(1:n_Nus, outer=6)
     #y = repeat(1:n_Nys, inner=5)
     #scatter!(x, y, z=vec(rmses_mean), yerr=vec(rmses_std), color=:red, ms=4, lc=:red, label=nothing)
     return p
end

end # module
