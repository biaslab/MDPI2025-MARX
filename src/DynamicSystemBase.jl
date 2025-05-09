module DynamicSystemBase

using Distributions # MvNormal
using ..Utils
using ..SystemBase
import ..SystemBase: get_eom_md, state_transition!, measure

# Define abstract types or interfaces for systems
abstract type DynamicSystem <: System end

export DynamicSystem

# Define shared interface functions as abstract
export ddzdt, inv_inertia_matrix, update_params

function state_transition!(sys::DynamicSystem, u::Vector{Float64})
    check_nan_or_inf(sys.z, "sys.z contains NaN or Inf: $(sys.z)")

    D_z = get_state_dim(typeof(sys))
    # TODO: assert D_z mod 2 ≡ 0
    D_zh = div(D_z,2)

    z, dz = sys.z[1:D_zh], sys.z[D_zh+1:end]
    z_new = z + sys.Δt * dz
    dz_new = dz + sys.Δt * ddzdt(sys, z, dz, u)

    sys.z = vcat(z_new, dz_new)

    update_params(sys)

    return sys.z
end

function measure(sys::DynamicSystem)
    D_y = get_observation_dim(typeof(sys))
    z = sys.z[1:2]
    y = rand(MvNormal(z, inv(sys.W)))
    return y
end

# ----------------------------
# Abstract Interface Functions
# ----------------------------
get_eom_md(sys::DynamicSystem) = not_implemented_error(typeof(sys))
ddzdt(sys::DynamicSystem, z::Vector{Float64}, dz::Vector{Float64}, u::Vector{Float64}) = not_implemented_error(typeof(sys))
inv_inertia_matrix(sys::DynamicSystem) = not_implemented_error(typeof(sys))
update_params(sys::DynamicSystem) = not_implemented_error(typeof(sys))

end
