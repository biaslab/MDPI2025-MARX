module MassSpringSystems

using LinearAlgebra
using Markdown

using ..Utils
using ..DynamicSystemBase
import ..DynamicSystemBase: get_eom_md, ddzdt, inv_inertia_matrix, update_params
import ..SystemBase: state_transition!, measure, get_eom_md, get_state_dim, get_observation_dim, get_action_dim, get_control_dim

export DoubleMassSpringDamperSystem

abstract type MassSpringSystem <: DynamicSystem end

const DEFAULTS_SYS = (
    DoubleMassSpringDamperSystem = (
        D_z = 4::Int,
        D_y = 2::Int,
        D_a = 2::Int,
        D_u = 2::Int,
        Δt = 0.01::Float64, # time step duration in s
        N = 500::Int, # number of episode steps
        z = [0.0, 0.0, 0.0, 0.0]::Vector{Float64}, # initial (hidden) state: z1, z2, dz1, dz2. wall should be at (0,0)
        W = 1e4*[0.2 0.1; 0.1 0.2]::Matrix{Float64}, # precision matrix
        mass = [1.0, 2.0]::Vector{Float64}, # ∀ cart: mass
        spring = [0.99, 0.8]::Vector{Float64}, # 1.0*ones(2)::Vector{Float64}, # ∀ cart: spring coefficient
        damping = [0.4, 0.4]::Vector{Float64}, # 0.1*ones(2)::Vector{Float64}, # ∀ cart: damping coefficient
        ubound = 0.1::Float64 # bounds for the control limits
    ),
)

mutable struct DoubleMassSpringDamperSystem <: MassSpringSystem
    # time
    k           ::Integer # environment lifetime timer
    Δt          ::Float64 # state transition step size
    N           ::Integer # maximum number of timesteps
    # state + observation noise
    z           ::Vector{Float64}
    W           ::Matrix{Float64}
    # parameters
    mass        ::Vector{Float64}
    spring      ::Vector{Float64} # [k1, k2]
    damping     ::Vector{Float64} # [c1, c2]

    function DoubleMassSpringDamperSystem(;
        N::Int=DEFAULTS_SYS.DoubleMassSpringDamperSystem.N,
        Δt::Float64=DEFAULTS_SYS.DoubleMassSpringDamperSystem.Δt,
        z::Vector{Float64}=DEFAULTS_SYS.DoubleMassSpringDamperSystem.z,
        mass::Vector{Float64}=DEFAULTS_SYS.DoubleMassSpringDamperSystem.mass,
        spring::Vector{Float64}=DEFAULTS_SYS.DoubleMassSpringDamperSystem.spring,
        damping::Vector{Float64}=DEFAULTS_SYS.DoubleMassSpringDamperSystem.damping,
        W::Matrix{Float64}=DEFAULTS_SYS.DoubleMassSpringDamperSystem.W
    )
        return new(1, Δt, N, z, W, mass, spring, damping)
    end
end

# ------------------------
# System Query Functions
# ------------------------

get_state_dim(sys_type::Type{DoubleMassSpringDamperSystem}) = DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_z
get_observation_dim(sys_type::Type{DoubleMassSpringDamperSystem}) = DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_y
get_action_dim(sys_type::Type{DoubleMassSpringDamperSystem}) = DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_a
get_control_dim(sys_type::Type{DoubleMassSpringDamperSystem}) = DEFAULTS_SYS.DoubleMassSpringDamperSystem.D_u

function get_eom_md(sys::DoubleMassSpringDamperSystem)
    return md"""
    # System dynamics
    $$\begin{aligned}
    \begin{bmatrix}
    m_1 & 0 \\
    0 & m_2
    \end{bmatrix}
    \begin{bmatrix} \ddot{z}_1 \\ \ddot{z}_2 \end{bmatrix} =
    \begin{bmatrix}
    - (c_1 + c_2) &   c_2 \\
               c_2  & - c_2
    \end{bmatrix}
    \begin{bmatrix} \dot{z}_1 \\ \dot{z}_2 \end{bmatrix}
    +
    \begin{bmatrix}
    - (k_1 + k_2) &     k_2 \\
                 k_2  & - k_2
    \end{bmatrix}
    \begin{bmatrix} z_1 \\ z_2 \end{bmatrix}
    +
    \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}
    \end{aligned}$$
    """
end

# ------------------------
# System Dynamics
# ------------------------

inv_inertia_matrix(sys::DoubleMassSpringDamperSystem) = diagm(1 ./ sys.mass)

function ddzdt(sys::DoubleMassSpringDamperSystem, z::Vector{Float64}, dz::Vector{Float64}, u::Vector{Float64})
    m1, m2 = sys.mass
    k1, k2 = sys.spring
    c1, c2 = sys.damping

    M = diagm([m1, m2])
    K = [-sum([k1, k2]) k2; k2 -k2]
    C = [-sum([c1, c2]) c2; c2 -c2]

    Mi = inv(M)
    MiK = Mi*K
    MiC = Mi*C

    return MiK * z + MiC * dz + Mi * u
end

function update_params(sys::DoubleMassSpringDamperSystem)
    return # do nothing
end

end # module
