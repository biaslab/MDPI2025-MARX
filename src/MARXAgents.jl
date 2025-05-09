module MARXAgents

using ..RingBuffers
using ..AgentBase
import ..AgentBase: params, predict, update!
using ..InteractionRecorder
using ..Utils

using Distributions # Normal etc
using LinearAlgebra # diagm
using SpecialFunctions # loggamma

export AbstractMARXAgent
export MARXAgent, HMARXAgent
export get_estimate_A, get_estimate_W
export pdf_params, pdf_predictive, pdf_likelihood
export surprisal, variational_free_energy
export ce_posterior_prior, e_posterior, ce_posterior_likelihood
export kl_posterior_prior, kl_posterior_likelihood
export update_params

const DEFAULTS = (
    T = 1::Int, # prediction horizon
    N_y = 2::Int, # memory buffer size, observations
    N_u = 2::Int, # memory buffer size, controls
    νadd = 0::Int, # degrees of freedom addition for Wishart: ν = D_y + νadd
    does_learn = true::Bool,
    does_dream = false::Bool,
    memory_type = "dim_first"::String # options: [yb_ub, dim_first]
)

abstract type AbstractMARXAgent <: OnlineProbabilisticAgent end

# TODO: ν can be a real (so change Int to Float)
mutable struct MARXAgent <: AbstractMARXAgent
    ID              ::Integer # agent ID
    # time
    k               ::Integer # agent lifetime timer
    T               ::Integer # planning horizon
    # dimension info
    D_y             ::Integer # dimension of observation space
    D_u             ::Integer # dimension of control space
    # memory
    D_x             ::Integer # memory size, total
    N_y             ::Integer # memory size, observations
    N_u             ::Integer # memory size, controls
    memory_type     ::String
    ybuffer         ::RingBuffer # memory, observations
    ubuffer         ::RingBuffer # memory, controls
    # parameters (MatrixNormal-Wishart likelihood) D_y: dimension of observation, D_x = N_y * D_y: dimension of state (result of M * memory_buffer)
    ν               ::Int # degrees of freedom for Wishart
    Ω               ::Matrix{Float64} # scale matrix of Wishart (size: D_y × D_y)
    Λ               ::Matrix{Float64} # row matrix of MatrixNormal (size: D_x × D_x)
    M               ::Matrix{Float64} # mean of MatrixNormal (size: D_x × D_y)
    # controls
    control_lims    ::Tuple{Float64, Float64} # control space range
    # switches
    does_learn      ::Bool
    does_dream      ::Bool
    # optional
    us_matlab       ::Union{Matrix{Float64}, Nothing}

    function MARXAgent(
        ID::Int,
        D_y::Int,
        D_u::Int,
        control_lims::Tuple{Float64, Float64};
        T::Int=DEFAULTS.T,
        N_y::Int=DEFAULTS.N_y,
        N_u::Int=DEFAULTS.N_u,
        cΩ::Float64=1e-2,
        cΛ::Float64=1e-1,
        νadd::Int=DEFAULTS.νadd,
        does_learn::Bool=DEFAULTS.does_learn,
        does_dream::Bool=DEFAULTS.does_dream,
        memory_type::String=DEFAULTS.memory_type
    )
        # MEMORY
        ybuffer = initialize_buffer(N_y + 1, D_y)
        ubuffer = initialize_buffer(N_u, D_u)
        D_x = compute_memory_size(N_y, D_y, N_u, D_u)
        @assert D_x > 0 "D_x = N_y * D_y ($N_y * $D_y) + N_u * D_u ($N_u * $D_u) must be > 0."

        # PARAMETERS
        ν = D_y + νadd
        Λ = cΛ*I(D_x)
        Ω = cΩ*I(D_y)
        M = zeros(D_y, D_x)

        return new(
            ID, 1, T,
            D_y, D_u,
            D_x, N_y, N_u, memory_type, ybuffer, ubuffer,
            ν, Ω, Λ, M,
            control_lims, does_learn, does_dream, nothing
        )
    end
end

mutable struct HMARXAgent <: AbstractMARXAgent
    ID              ::Integer # agent ID
    # time
    k               ::Integer # agent lifetime timer
    T               ::Integer # planning horizon
    # dimension info
    D_y             ::Integer # dimension of observation space
    D_u             ::Integer # dimension of control space
    # memory
    D_x             ::Integer # memory size, total
    N_y             ::Integer # memory size, observations
    N_u             ::Integer # memory size, controls
    memory_type     ::String
    ybuffer         ::RingBuffer # memory, observations
    ubuffer         ::RingBuffer # memory, controls
    # parameters (MatrixNormal-Wishart likelihood) D_y: dimension of observation, D_x = N_y * D_y: dimension of state (result of M * memory_buffer)
    ν               ::Int # degrees of freedom for Wishart
    Ω               ::Matrix{Float64} # scale matrix of Wishart (size: D_y × D_y)
    Λ               ::Matrix{Float64} # row matrix of MatrixNormal (size: D_x × D_x)
    M               ::Matrix{Float64} # mean of MatrixNormal (size: D_x × D_y)
    # controls
    control_lims    ::Tuple{Float64, Float64} # control space range
    # switches
    does_learn      ::Bool
    does_dream      ::Bool
    # optional
    us_matlab       ::Union{Matrix{Float64}, Nothing}

    ν_w::Int
    Ω_w::Matrix{Float64}
    Λ_w::Matrix{Float64}
    M_w::Matrix{Float64}

    function HMARXAgent(
        ID::Int,
        D_y::Int,
        D_u::Int,
        control_lims::Tuple{Float64, Float64};
        T::Int = DEFAULTS.T,
        N_y::Int = DEFAULTS.N_y,
        N_u::Int = DEFAULTS.N_u,
        cΩ::Float64 = 1e-2,
        cΛ::Float64 = 1e-1,
        cΩ_w::Float64 = 1e-2,
        cΛ_w::Float64 = 1e-1,
        νadd::Int = DEFAULTS.νadd,
        νadd_w::Int = DEFAULTS.νadd,
        does_learn::Bool = DEFAULTS.does_learn,
        does_dream::Bool = DEFAULTS.does_dream
    )
        # Initialize base MARXAgent fields
        parent = MARXAgent(ID, D_y, D_u, control_lims; T=T, N_y=N_y, N_u=N_u, cΩ=cΩ, cΛ=cΛ, νadd=νadd, does_learn=does_learn, does_dream=does_dream)

        # Initialize HMARXAgent-specific fields
        ν_w = parent.D_y + νadd
        Λ_w = cΛ_w * I(parent.D_x)
        Ω_w = cΩ_w * I(parent.D_y)
        M_w = zeros(parent.D_y, parent.D_x)

        # Create new HMARXAgent with parent fields and added fields
        return new(
            parent.ID, parent.k, parent.T,
            parent.D_y, parent.D_u, parent.D_x,
            parent.N_y, parent.N_u, parent.memory_type,
            parent.ybuffer, parent.ubuffer,
            parent.ν, parent.Ω, parent.Λ, parent.M,
            parent.control_lims, parent.does_learn, parent.does_dream, parent.us_matlab,
            ν_w, Ω_w, Λ_w, M_w
        )
    end
end

params(agent::AbstractMARXAgent) = agent.ν, agent.Λ, agent.Ω, agent.M

function update_params(agent::MARXAgent)
    ν0, Λ0, Ω0, M0 = params(agent)
    x = memory(agent)
    y = get_last(agent.ybuffer)

    ν = ν0 + 1
    Λ = Λ0 + x*x'
    M = (M0*Λ0 + y*x')*inv(Λ)
    Ω = Ω0 + y*y' + M0*Λ0*M0' - M*Λ*M'
    return ν, Λ, Ω, M
end

function update!(agent::MARXAgent)
    ν0, Λ0, Ω0, M0 = params(agent)
    x = memory(agent)
    y = get_last(agent.ybuffer)

    agent.ν += 1
    agent.Λ += x*x'
    agent.M = (M0*Λ0 + y*x')*inv(agent.Λ)
    agent.Ω += y*y' + M0*Λ0*M0' - agent.M*agent.Λ*agent.M'
end

# TODO: Update with addition node
function update!(agent::HMARXAgent)
    ν0, Λ0, Ω0, M0 = params(agent)
    x = memory(agent)
    y = get_last(agent.ybuffer)

    # MARX posterior
    agent.ν += 1
    agent.Λ += x*x'
    agent.M = (M0*Λ0 + y*x')*inv(agent.Λ)
    agent.Ω += y*y' + M0*Λ0*M0' - agent.M*agent.Λ*agent.M'

    ν1, Λ1, Ω1, M1 = params(agent)

    # Additional random walk
    agent.ν += agent.ν_w + agent.D_x - agent.D_y - 1
    agent.Λ += agent.Λ_w
    agent.M = M1 * Λ1 * inv(agent.Λ)
    agent.Ω += agent.Ω_w + M1*Λ1*M1' - agent.M*agent.Λ*agent.M'

    # TODO: update ω params
end

get_estimate_A(agent::AbstractMARXAgent) = agent.M
get_estimate_W(agent::AbstractMARXAgent) = agent.ν * inv(agent.Ω)

# online prediction
function predict(agent::AbstractMARXAgent)
    @assert agent.ν > 2 "covariance of posterior predictive is undefined for ν <= 2. Current ν: $(agent.ν)"
    x = memory(agent)
    μ = agent.M*x
    η = agent.ν - agent.D_y + 1
    cΣ = (1 + x'*inv(agent.Λ)*x)/η
    Σ = cΣ * agent.Ω
    y_Σ = (η / (η - 2)) * Σ
    return μ, y_Σ
end

# offline prediction
function predict(agent::AbstractMARXAgent, rec::Recorder)
    (D_x, N) = size(rec.xs)
    μ = agent.M*rec.xs
    y_Σ = zeros(agent.D_y, agent.D_y, N)
    for t in 1:N
        x = rec.xs[:,t]
        η = agent.ν - agent.D_y + 1
        cΣ = (1 + x'*inv(agent.Λ)*x)/η
        Σ = cΣ * agent.Ω
        y_Σ[:,:,t] .= (η / (η - 2)) * Σ
    end
    return μ, y_Σ
end

function pdf_params(agent::AbstractMARXAgent, A::Matrix{Float64}, W::Matrix{Float64}; logpdf::Bool=true)
    ν, Λ, Ω, M = params(agent)
    diff = A-M
    log_pdf_value = 0.5 * tr(W*diff*Λ*diff' + Ω)
    return log_pdf_value # returning already negated
end

function pdf_params(A::Matrix{Float64}, W::Matrix{Float64}, M::Matrix{Float64}, Λ::Matrix{Float64}, Ω::Matrix{Float64}, ν::Float64, D_x::Int, D_y::Int; logpdf::Bool=true)
    diff = A - M
    exp_term = -0.5 * tr(W * diff * Λ * diff' + Ω)
    normalizing_term = 0
    normalizing_term -= 0.5 * ν * D_y * log(2)
    normalizing_term -= 0.5 * D_x * D_y * log(2π)
    normalizing_term += 0.5 * D_y * log(det(Ω))
    normalizing_term += 0.5 * ν * log(det(Λ))
    normalizing_term += 0.5 * (ν + D_x - D_y - 1) * log(det(W))
    normalizing_term -= 0.5 * logmvgamma(D_y, 0.5 * ν)

    log_pdf_value = normalizing_term + exp_term

    return -log_pdf_value
end

function pdf_predictive(y::Vector{Float64}, x::Vector{Float64}, M::Matrix{Float64}, Λ::Matrix{Float64}, Ω::Matrix{Float64}, ν::Float64, D_y::Int;logpdf::Bool=true)
    η = ν - D_y + 1
    λ = (1 + x'*inv(Λ)*x)
    μ = M*x
    psi = η*inv(Ω)*λ
    diff = y - μ
    if !logpdf
        pdf_value = 1
        pdf_value *= det(psi)^(0.5)
        pdf_value *= (η*π)^(-0.5*D_y)
        pdf_value *= mvgamma(D_y, 0.5*(η + D_y))
        pdf_value /= mvgamma(D_y, 0.5*(η + D_y - 1))
        pdf_value *= (1 + (diff' * psi * diff)/η)^(-0.5*(η + D_y))
    else
        pdf_value = 0
        pdf_value += 0.5*log(det(psi))
        pdf_value -= (0.5*D_y)*log(η*π)
        pdf_value += logmvgamma(D_y, 0.5 * (η + D_y))
        pdf_value -= logmvgamma(D_y, 0.5 * (η + D_y - 1))
        pdf_value -= 0.5*(η + D_y) * log(1 + (diff' * psi * diff)/η)
    end
    return pdf_value
end

function pdf_likelihood(y::Vector{Float64}, x::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64}, D_y::Int; logpdf::Bool=true)
    m = A*x
    diff = y - m
    if !logpdf
        pdf_value = 1
        # TODO
    else
        pdf_value = 0
        pdf_value += 0.5 * log(det(W))
        pdf_value -= 0.5 * D_y * log(2*π)
        pdf_value -= 0.5 * diff' * W * diff
    end
    return pdf_value
end

# this function needs to be called AFTER the agent observes the new observation (s.t. get_last gets the current observation and x contains the previous observation) and BEFORE it updates its parameters
# NOTE: M = M' and A = A' since we have calculate the transpose in code
function surprisal(agent::AbstractMARXAgent)
    ν, Λ, Ω, M = params(agent)
    D_y = agent.D_y
    y = get_last(agent.ybuffer)
    x = memory(agent)
    λ = 1/(1 + x'*inv(Λ)*x)
    diff = y - M*x

    #println(-D_y*log(λ))

    value = 0.0
    value += D_y*log(π)
    value -= D_y*log(λ)
    value += log(det(Ω))
    value -= 2*logmvgamma(D_y, 0.5*(ν+1))
    value += 2*logmvgamma(D_y, 0.5*ν)
    value += (ν+1)*log(1 + λ*diff'*inv(Ω)*diff)
    value *= 0.5
    return value
end

function ce_mnw_mnw(ν_q::Int, Λ_q::Matrix{Float64}, Ω_q::Matrix{Float64}, M_q::Matrix{Float64}, ν_p::Int, Λ_p::Matrix{Float64}, Ω_p::Matrix{Float64}, M_p::Matrix{Float64}, D_y::Int, D_x::Int)
    #println(D_y*tr(inv(Λ_q)*Λ_p'))
    #println()

    diff = M_q - M_p
    value = 0.0
    value -= D_y*log(det(Λ_p))
    value += (ν_p + D_x - D_y - 1)*log(det(Ω_q))
    value -= ν_p*log(det(Ω_p))
    value += (D_y + 1)*D_y*log(2)
    value += D_x*D_y*log(π)
    value += 2*logmvgamma(D_y, 0.5*ν_p)
    value -= (ν_p + D_x - D_y - 1)*mvdigamma(D_y, 0.5*ν_q)
    value += ν_q*tr(inv(Ω_q)*diff*Λ_p*diff')
    value += ν_q*tr(inv(Ω_q)*Ω_p)
    value += D_y*tr(inv(Λ_q)*Λ_p')
    value *= 0.5
    return value
end

# TODO: e_mwn for faster calculation
#
function ce_mnw_mvn(ν_q::Int, Λ_q::Matrix{Float64}, Ω_q::Matrix{Float64}, M_q::Matrix{Float64}, y::Vector{Float64}, x::Vector{Float64}, D_y::Int)
    #println(D_y*x'inv(Λ_q)*x)
    diff = y - M_q*x
    value = 0.0
    value -= mvdigamma(D_y, 0.5*ν_q)
    value += log(det(Ω_q))
    value += D_y*log(π)
    value += ν_q*diff'*inv(Ω_q)*diff
    value += D_y*x'inv(Λ_q)*x
    value *= 0.5
    return value
end

function ce_posterior_prior(agent::AbstractMARXAgent)
    ν_p, Λ_p, Ω_p, M_p = params(agent)
    ν_q, Λ_q, Ω_q, M_q = update_params(agent)
    return ce_mnw_mnw(ν_q, Λ_q, Ω_q, M_q, ν_p, Λ_p, Ω_p, M_p, agent.D_y, agent.D_x)
end

function e_posterior(agent::AbstractMARXAgent)
    ν_q, Λ_q, Ω_q, M_q = update_params(agent)
    return ce_mnw_mnw(ν_q, Λ_q, Ω_q, M_q, ν_q, Λ_q, Ω_q, M_q, agent.D_y, agent.D_x)
end

function ce_posterior_likelihood(agent::AbstractMARXAgent)
    ν_q, Λ_q, Ω_q, M_q = update_params(agent)
    y = get_last(agent.ybuffer)
    x = memory(agent)
    return ce_mnw_mvn(ν_q, Λ_q, Ω_q, M_q, y, x, agent.D_y)
end

# KL[q||p] = H[q,p] - H[q]
function kl_posterior_prior(agent::AbstractMARXAgent)
    ν_p, Λ_p, Ω_p, M_p = params(agent)
    ν_q, Λ_q, Ω_q, M_q = update_params(agent)
    centropy = ce_mnw_mnw(ν_q, Λ_q, Ω_q, M_q, ν_p, Λ_p, Ω_p, M_p, agent.D_y, agent.D_x)
    sentropy = ce_mnw_mnw(ν_q, Λ_q, Ω_q, M_q, ν_q, Λ_q, Ω_q, M_q, agent.D_y, agent.D_x)
    return centropy - sentropy
end

function kl_posterior_likelihood(agent::AbstractMARXAgent)
    ν_p, Λ_p, Ω_p, M_p = params(agent)
    ν_q, Λ_q, Ω_q, M_q = update_params(agent)
    y = get_last(agent.ybuffer)
    x = memory(agent)
    centropy = ce_mnw_mvn(ν_q, Λ_q, Ω_q, M_q, y, x, agent.D_y)
    sentropy = ce_mnw_mnw(ν_q, Λ_q, Ω_q, M_q, ν_q, Λ_q, Ω_q, M_q, agent.D_y, agent.D_x)
    return centropy - sentropy
end

function variational_free_energy(agent::AbstractMARXAgent)
    return ce_posterior_prior(agent) + ce_posterior_likelihood(agent) - e_posterior(agent) # should be the same as surprisal(agent)
end

end
