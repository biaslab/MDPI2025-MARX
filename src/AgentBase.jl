module AgentBase

using ..RingBuffers
using ..Utils
using ..InteractionRecorder

export Agent, OnlineDeterministicAgent, OfflineDeterministicAgent, OnlineProbabilisticAgent, OfflineProbabilisticAgent
# Implemented functions
export set_learning!, set_dreaming!, override_controls!, observe!, memorize_observation!, memory, trigger_amnesia!, decide!, memorize_control!, react!
# To be implemented by inheriting module
export params, predict, update!

abstract type Agent end
abstract type OnlineDeterministicAgent <: Agent end
abstract type OfflineDeterministicAgent <: Agent end
abstract type OnlineProbabilisticAgent <: Agent end
abstract type OfflineProbabilisticAgent <: Agent end

set_learning!(agent::Agent, does_learn::Bool) = agent.does_learn = does_learn
set_dreaming!(agent::Agent, does_dream::Bool) = agent.does_dream = does_dream

function override_controls!(agent::Agent, N_fmultisingen::Int, P_fmultisingen::Int, t_beg::Int, t_end::Int; a_scale::Float64=1.0)
    N = N_fmultisingen*P_fmultisingen
    agent.k = 1
    us = load_generated_controls("data/generated/uTrain.mat")
    agent.us_matlab = reshape(us[t_beg:t_end], (agent.D_y, N))
end

function decide!(agent::Agent)
    u = agent.us_matlab[:, agent.k]
    memorize_control!(agent, u)
    agent.k += 1
    return u
end

# observe a new observation by memorization using our preferred memory (in this case a ring buffer)
observe!(agent::Agent, y::Vector{Float64}) = memorize_observation!(agent, y)

function memorize_observation!(agent::Agent, y::Vector{Float64})
    agent.N_y > 0 && push!(agent.ybuffer, y)
end

function memorize_control!(agent::Agent, u::Vector{Float64})
    agent.N_u > 0 && push!(agent.ubuffer, u)
end


function memory(agent::Agent)
    yb, ub = agent.ybuffer, agent.ubuffer
    x = RingBuffer(agent.D_x, Float64)

    if agent.memory_type == "dim_first"
        ys = reshape(get_vector(yb), (agent.D_y, agent.N_y + 1))
        us = reshape(get_vector(ub), (agent.D_u, agent.N_u))
        for dim in 1:agent.D_y
            push!(x, ys[dim, 1:agent.N_y]...)
        end
        for dim in 1:agent.D_u
            push!(x, us[dim, :]...)
        end
    elseif agent.memory_type == "yb_ub"
        agent.N_y > 0 && push!(x, get_vector(yb)[1:agent.D_y*agent.N_y]...)
        agent.N_u > 0 && push!(x, get_vector(ub)...)
    else
        throw(ArgumentError("Unsupported memory type: $memory_type"))
    end
    return get_elements(x)
end

function trigger_amnesia!(agent::Agent)
    agent.ybuffer = initialize_buffer(agent.N_y + 1, agent.D_y)
    agent.ubuffer = initialize_buffer(agent.N_u, agent.D_u)
end

# reaction of an agent upon observing an observation y at current time step k that triggers multiple things
function react!(agent::Agent, y::Vector{Float64})
    m, Σ = agent isa OnlineProbabilisticAgent ? predict(agent) : (predict(agent), nothing)
    observe!(agent, agent.does_dream ? m : y)
    if agent.does_learn update!(agent) end
    return decide!(agent)
end

# TO IMPLEMENT

params(agent::Agent) = not_implemented_error(agent)
update!(agent::OnlineDeterministicAgent) = not_implemented_error(agent)
update!(agent::OnlineProbabilisticAgent) = not_implemented_error(agent)
predict(agent::OnlineDeterministicAgent) = not_implemented_error(agent)
predict(agent::OnlineProbabilisticAgent) = not_implemented_error(agent)
predict(agent::OfflineProbabilisticAgent, rec::Recorder) = not_implemented_error(agent)

end # module
