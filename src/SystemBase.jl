module SystemBase

using ..Utils

export System
# implemented functions
export step!
export get_realtime_xticks, get_state_dim, get_observation_dim, get_action_dim, get_control_dim
export get_control, get_state, get_tsteps, convert_action
# need to be implemented by inheriting module
export get_eom_md, state_transition!, measure

abstract type System end

# -------------------
# Core System Functions
# -------------------

function step!(sys::System, a::Vector{Float64}, ulims::Tuple{Float64, Float64})
    update!(sys, a, ulims)
    sys.k += 1
end

function update!(sys::System, a::Vector{Float64}, ulims::Tuple{Float64, Float64})
    u = convert_action(a, ulims)
    state_transition!(sys, u)
end

# -------------------
# Action and Time Management
# -------------------

convert_action(a::Vector{Float64}, ulims::Tuple{Float64, Float64}) = clamp.(a, ulims...)
get_tsteps(sys::System) = range(0, step=sys.Δt, length=sys.N)

function get_realtime_xticks(sys::System)
    xticks_pos = collect(0:(sys.N/5):sys.N) .* sys.Δt
    xticks_labels = string.(Int.(round.(xticks_pos)))
    return xticks_pos, xticks_labels
end


# -------------------
# State and Control Queries
# -------------------

get_control(sys::System) = sys.torque
get_state(sys::System) = sys.z

# -------------------
# Not Implemented Placeholders
# -------------------

get_state_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_observation_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_action_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_control_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_eom_md(sys::System) = not_implemented_error(typeof(sys))
state_transition!(sys::System, u::Vector{Float64}) = not_implemented_error(typeof(sys))
measure(sys::System) = not_implemented_error(typeof(sys))

end # module
