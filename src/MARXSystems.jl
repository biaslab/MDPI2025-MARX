module MARXSystems

using LinearAlgebra
using Markdown
using Distributions

using ..Utils
import ..SystemBase: System, state_transition!, measure, get_eom_md, get_state_dim, get_observation_dim, get_action_dim, get_control_dim
using ..RingBuffers

export AbstractMARXSystem, MARXSystem, TvMARXSystem
export get_state_transition_matrix, get_process_noise_matrix

abstract type AbstractMARXSystem <: System end

const DEFAULTS_SYS = (
    MARXSystem = (
        D_z = 2::Int,
        D_y = 2::Int,
        D_a = 2::Int,
        D_u = 2::Int,
        N_y = 2::Int,
        N_u = 3::Int,
        Δt = 1.00::Float64, # time step duration in s
        N = 500::Int, # number of episode steps
        z = [0.0, 0.0]::Vector{Float64}, # initial (hidden) state: z1, z2, dz1, dz2. wall should be at (0,0)
        W = 1e3*[0.3 0.1; 0.1 0.2]::Matrix{Float64}, # precision matrix
        ubound = 0.1::Float64, # bounds for the control limits
        memory_type = "dim_first"::String
    ),
)

# time-invariant parameters
mutable struct MARXSystem <: AbstractMARXSystem
    # time
    k           ::Integer # environment lifetime timer
    Δt          ::Float64 # state transition step size
    N           ::Integer # maximum number of timesteps
    # state + observation noise
    z           ::Vector{Float64}
    W           ::Matrix{Float64}
    # parameters
    A           ::Matrix{Float64}
    # memory
    D_x             ::Integer
    N_y             ::Integer # memory size, observations
    N_u             ::Integer # memory size, controls
    memory_type     ::String
    ybuffer         ::RingBuffer # memory, observations
    ubuffer         ::RingBuffer # memory, controls

    function MARXSystem(;
        N::Int=DEFAULTS_SYS.MARXSystem.N,
        Δt::Float64=DEFAULTS_SYS.MARXSystem.Δt,
        z::Vector{Float64}=DEFAULTS_SYS.MARXSystem.z,
        W::Matrix{Float64}=DEFAULTS_SYS.MARXSystem.W,
        memory_type::String=DEFAULTS_SYS.MARXSystem.memory_type
    )
        N_y, N_u = DEFAULTS_SYS.MARXSystem.N_y, DEFAULTS_SYS.MARXSystem.N_u
        D_y, D_u = DEFAULTS_SYS.MARXSystem.D_y, DEFAULTS_SYS.MARXSystem.D_u

        ybuffer = initialize_buffer(N_y, D_y)
        ubuffer = initialize_buffer(N_u, D_u)
        D_x = compute_memory_size(N_y, D_y, N_u, D_u)
        @assert D_x > 0 "D_x = N_y * D_y ($N_y * $D_y) + N_u * D_u ($N_u * $D_u) must be > 0."

        A = setup_system_matrix(memory_type, D_y, D_u, D_x, N_y, N_u)
        return new(1, Δt, N, z, W, A, D_x, N_y, N_u, memory_type, ybuffer, ubuffer)
    end
end

# time-variant parameters
mutable struct TvMARXSystem <: AbstractMARXSystem
    # time
    k           ::Integer # environment lifetime timer
    Δt          ::Float64 # state transition step size
    N           ::Integer # maximum number of timesteps
    # state + observation noise
    z           ::Vector{Float64}
    W           ::Matrix{Float64}
    # parameters
    A           ::Matrix{Float64}
    # memory
    D_x             ::Integer
    N_y             ::Integer # memory size, observations
    N_u             ::Integer # memory size, controls
    memory_type     ::String
    ybuffer         ::RingBuffer # memory, observations
    ubuffer         ::RingBuffer # memory, controls

    function TvMARXSystem(;
        N::Int=DEFAULTS_SYS.MARXSystem.N,
        Δt::Float64=DEFAULTS_SYS.MARXSystem.Δt,
        z::Vector{Float64}=DEFAULTS_SYS.MARXSystem.z,
        W::Matrix{Float64}=DEFAULTS_SYS.MARXSystem.W,
        memory_type::String=DEFAULTS_SYS.MARXSystem.memory_type
    )
        parent = MARXSystem(;N=N, Δt=Δt, z=z, W=W, memory_type=memory_type)
        return new(parent.k, parent.Δt, parent.N, parent.z, parent.W, parent.A, parent.D_x, parent.N_y, parent.N_u, parent.memory_type, parent.ybuffer, parent.ubuffer)
    end
end

# ------------------------
# System Query Functions
# ------------------------

get_state_dim(sys_type::Type{<:AbstractMARXSystem}) = DEFAULTS_SYS.MARXSystem.D_z
get_observation_dim(sys_type::Type{<:AbstractMARXSystem}) = DEFAULTS_SYS.MARXSystem.D_y
get_action_dim(sys_type::Type{<:AbstractMARXSystem}) = DEFAULTS_SYS.MARXSystem.D_a
get_control_dim(sys_type::Type{<:AbstractMARXSystem}) = DEFAULTS_SYS.MARXSystem.D_u

function get_eom_md(sys::AbstractMARXSystem)
    return md"""
    # System dynamics
    $$\begin{aligned}\begin{bmatrix} y_{1,k} \\ y_{2,k} \end{bmatrix} = A \bar{y}_{k-1} + e_k \quad , \quad e_k \sim \mathcal{N}(0, W^{-1})\end{aligned}$$
    """
end

get_state_transition_matrix(sys::C) where {C <: AbstractMARXSystem} = sys.A
get_process_noise_matrix(sys::C) where {C <: AbstractMARXSystem} = sys.W

# ------------------------
# System Memory Functions
# ------------------------

function memorize_observation!(sys::T, y::Vector{Float64}) where {T <: AbstractMARXSystem}
    sys.N_y > 0 && push!(sys.ybuffer, y)
end

function memorize_control!(sys::T, u::Vector{Float64}) where {T <: AbstractMARXSystem}
    sys.N_u > 0 && push!(sys.ubuffer, u)
end

function memory_system(sys::T) where {T <: AbstractMARXSystem}
    yb, ub = sys.ybuffer, sys.ubuffer
    D_y, N_y, D_u, N_u, D_x = DEFAULTS_SYS.MARXSystem.D_y, sys.N_y, DEFAULTS_SYS.MARXSystem.D_u, sys.N_u, sys.D_x

    x = RingBuffer(D_x, Float64)
    if sys.memory_type == "dim_first"
        ys = reshape(get_vector(yb), (D_y, N_y))
        us = reshape(get_vector(ub), (D_u, N_u))
        for dim in 1:D_y
            push!(x, ys[dim, 1:N_y]...)
        end
        for dim in 1:D_u
            push!(x, us[dim, :]...)
        end
    elseif sys.memory_type == "yb_ub"
        N_y > 0 && push!(x, get_vector(yb)[1:D_y*N_y]...)
        N_u > 0 && push!(x, get_vector(ub)...)
    else
        throw(ArgumentError("Unsupported memory type: $memory_type"))
    end
    x = get_elements(x)
    return x
end

# ------------------------
# System Dynamics & Measurement
# ------------------------

function state_transition!(sys::MARXSystem, u::Vector{Float64})
    memorize_control!(sys, u)
    x = memory_system(sys)
    sys.z = sys.A*x
    return sys.z
end

function state_transition!(sys::TvMARXSystem, u::Vector{Float64})
    memorize_control!(sys, u)
    x = memory_system(sys)
    sys.z = sys.A*x
    D_y = DEFAULTS_SYS.MARXSystem.D_y
    sys.A += 1e-2*randn(D_y, sys.D_x)
    return sys.z
end

function measure(sys::T) where {T <: AbstractMARXSystem}
    D_y = DEFAULTS_SYS.MARXSystem.D_y
    error = rand(MvNormal(zeros(D_y), inv(sys.W)))
    y = sys.z + error
    memorize_observation!(sys, sys.k == 1 ? zeros(D_y) : y) # TODO: without this special case: mismatch notebook vs this code
    return y
end

end # module
