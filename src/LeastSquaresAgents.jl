module LeastSquaresAgents

using ..RingBuffers
using ..Utils
using ..AgentBase
import ..AgentBase: params, predict, update!
using ..InteractionRecorder

using Distributions # Normal etc
using LinearAlgebra # diagm

export OfflineLeastSquaresAgent, OnlineLeastSquaresAgent

DEFAULTS = (
    T = 1::Int, # prediction horizon
    N_y = 2::Int, # memory buffer size, observations
    N_u = 3::Int, # memory buffer size, controls
    does_learn = true::Bool,
    does_dream = false::Bool,
    memory_type = "dim_first"::String
)

mutable struct OfflineLeastSquaresAgent <: OfflineDeterministicAgent
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
    M               ::Matrix{Float64} # mean of MatrixNormal (size: D_x × D_y)
    # controls
    control_lims    ::Tuple{Float64, Float64} # control space range
    # switches
    does_learn      ::Bool
    does_dream      ::Bool
    # optional
    us_matlab       ::Union{Matrix{Float64}, Nothing}

    function OfflineLeastSquaresAgent(
        ID::Int,
        D_y::Int,
        D_u::Int,
        control_lims::Tuple{Float64, Float64};
        T::Int=DEFAULTS.T,
        N_y::Int=DEFAULTS.N_y,
        N_u::Int=DEFAULTS.N_u,
        does_learn::Bool=DEFAULTS.does_learn,
        does_dream::Bool=DEFAULTS.does_dream,
        memory_type::String=DEFAULTS.memory_type
    )
        # MEMORY
        ybuffer = initialize_buffer(N_y + 1, D_y)
        ubuffer = initialize_buffer(N_u, D_u)
        D_x = compute_memory_size(N_y, D_y, N_u, D_u)
        @assert D_x > 0 "D_x = N_y * D_y ($N_y * $D_y) + N_u * D_u ($N_u * $D_u) must be > 0."

        M = zeros(D_y, D_x)

        return new(
            ID, 1, T,
            D_y, D_u,
            D_x, N_y, N_u, memory_type, ybuffer, ubuffer,
            M,
            control_lims, does_learn, does_dream, nothing
        )
    end
end

mutable struct OnlineLeastSquaresAgent <: OnlineDeterministicAgent
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
    M               ::Matrix{Float64} # mean of MatrixNormal (size: D_x × D_y)
    P               ::Vector{Matrix{Float64}} # covariance matrix representing confidence in the current estimate of M
    λ               ::Float64 # forgetting factor / regularization term: control how much weight is given to new data vs past data. slighlty below 1: gradual adapation to changing conditions
    δ               ::Float64 # initial covariance factor / initialization factor. high value = low confidence = better adaptation
    # controls
    control_lims    ::Tuple{Float64, Float64} # control space range
    # switches
    does_learn      ::Bool
    does_dream      ::Bool
    # optional
    us_matlab       ::Union{Matrix{Float64}, Nothing}

    function OnlineLeastSquaresAgent(
        ID::Int,
        D_y::Int,
        D_u::Int,
        control_lims::Tuple{Float64, Float64};
        T::Int=DEFAULTS.T,
        N_y::Int=DEFAULTS.N_y,
        N_u::Int=DEFAULTS.N_u,
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
        M = zeros(D_y, D_x)
        δ = 1.0
        P = [δ * I(D_x) for _ in 1:D_y]
        λ = 1.0

        return new(
            ID, 1, T,
            D_y, D_u,
            D_x, N_y, N_u, memory_type, ybuffer, ubuffer,
            M, P, λ, δ,
            control_lims, does_learn, does_dream, nothing
        )
    end
end

function params(agent::OfflineLeastSquaresAgent)
    return agent.M
end

function params(agent::OnlineLeastSquaresAgent)
    return agent.M, agent.P, agent.λ, agent.δ
end

function update!(agent::OfflineLeastSquaresAgent, rec::Recorder)
    agent.M = (rec.ys*rec.xs') / (rec.xs*rec.xs')
end

function update!(agent::OnlineLeastSquaresAgent)
    x = memory(agent)
    y = get_last(agent.ybuffer)
    M0, P0, λ = params(agent)

    for i in 1:agent.D_y
        K_i = P0[i] * x / (λ + x'*P0[i]*x)
        agent.M[i, :] += K_i*(y[i] - dot(M0[i,:], x))
        agent.P[i] = (P0[i] - K_i*x'*P0[i]) / λ
    end
end

function predict(agent::OfflineLeastSquaresAgent, rec::Recorder)
    return agent.M*rec.xs
end

function predict(agent::OnlineLeastSquaresAgent)
    return agent.M*memory(agent)
end

end
