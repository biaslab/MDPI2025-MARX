module InteractionRecorder

export Recorder
export record_zs!, record_ys!, record_as!, record_us!, record_xs!, record_νs!, record_Λs!, record_Ωs!, record_Ws!, record_Ms!, record_As!
export record_surprisals!, record_vfes!, record_ces_posterior_prior!, record_ces_posterior_likelihood!, record_es_posterior!, record_kls_posterior_prior!, record_kls_posterior_likelihood!
export record_prediction_mean!, record_prediction_covariance!

# TODO: Maybe different recorders for agent, environment, and interaction

# Define a struct to store observations and predicted means
struct Recorder
    zs::Array{Float64, 2}  # Stores states (D_z × N)
    ys::Array{Float64, 2}  # Stores observations (D_y × N)
    as::Array{Float64, 2}  # Stores actions (D_a × N)
    #us::Array{Float64, 2}  # Stores observations (D_u × N)
    #xs::Array{Float64, 2}  # Stores observations (D_x × N)
    #pred_ys::Array{Float64, 3}  # Stores predicted means (D_y × N)
    #pred_Σs::Array{Float64, 4}  # Stores predicted covariances (D_y × N)
    #νs::Array{Float64, 1}  # Stores predicted parameter means (D_y × N)
    #Λs::Array{Float64, 3}  # Stores predicted parameter means (D_x × D_x × N)
    #Ωs::Array{Float64, 3}  # Stores predicted parameter means (D_y × D_y × N)
    #Ws::Array{Float64, 3}  # Stores predicted parameter means (D_y × D_y × N)
    #Ms::Array{Float64, 3}  # Stores predicted parameter means (D_y × D_x × N)
    #As::Array{Float64, 3} # Stores system state transition matrix (D_y × D_x × N)
    #surprisals::Array{Float64, 1} # Stores surprisal values
    #vfes::Array{Float64, 1} # Stores variational free energy values
    #ces_posterior_prior::Array{Float64, 1} # Stores cross-entropies posterior relative to prior
    #ces_posterior_likelihood::Array{Float64, 1} # Stores cross-entropies posterior relative to likelihood
    #es_posterior::Array{Float64, 1} # Stores entropy of posterior
    #kls_posterior_prior::Array{Float64, 1} # Stores cross-entropies posterior relative to prior
    #kls_posterior_likelihood::Array{Float64, 1} # Stores cross-entropies posterior relative to likelihood
end

# Initialize the recorder with the correct dimensions
function Recorder(D_z::Int, D_y::Int, D_a::Int, D_u::Int, D_x::Int, N::Int, K::Int)
    return Recorder(
        zeros(D_z, N),       # zs
        zeros(D_y, N),       # ys
        zeros(D_a, N),       # as
        #zeros(D_u, N),       # us
        #zeros(D_x, N),       # xs
        #zeros(D_y, N, K),    # pred_ys
        #zeros(D_y, D_y, N, K), # pred_Σs
        #zeros(N),            # νs
        #zeros(D_x, D_x, N),  # Λs
        #zeros(D_y, D_y, N),  # Ωs
        #zeros(D_y, D_y, N),  # Ws
        #zeros(D_y, D_x, N),  # Ms
        #zeros(D_y, D_x, N),  # As
        #zeros(N),            # surprisals
        #zeros(N),            # VFE
        #zeros(N),            # H[q,p] posterior - prior
        #zeros(N),            # H[q,p] posterior - likelihood
        #zeros(N),            # H[q] posterior
        #zeros(N),            # KL[q||p] posterior - prior
        #zeros(N)             # KL[q||p] posterior - prior
    )
end

macro record_fn(field, dims)
    field_sym = Symbol(field)

    fn_args = :(recorder::Recorder, value, t::Int)
    fn_body =
        if dims == 1 quote recorder.$field_sym[t] = value end
        elseif dims == 2 quote recorder.$field_sym[:, t] .= value end
        elseif dims == 3 quote recorder.$field_sym[:, :, t] .= value end
        else error("Unsupported dimensionality: $dims")
        end

    fn_name = Symbol("record_", field, "!")
    fn = quote
        function $(esc(fn_name))(recorder::Recorder, value, t::Int)
            $(fn_body)
        end
    end
    return fn
end

macro record_fn_other(field, dims)
    quote
        function $(Symbol("record_", field, "!"))(recorder::Recorder, value, t::Int)
            @assert ndims(value) == $dims "Input value must have $dims dimensions for $field"
            recorder.$(Symbol(field))[CartesianIndices(value)..., t] .= value
        end
    end
end

@record_fn zs 2
@record_fn ys 2
@record_fn as 2
#@record_fn us 2
#@record_fn xs 2
#@record_fn νs 1
#@record_fn Λs 3
#@record_fn Ωs 3
#@record_fn Ws 3
#@record_fn Ms 3
#@record_fn As 3
#@record_fn surprisals 1
#@record_fn vfes 1
#@record_fn ces_posterior_prior 1
#@record_fn ces_posterior_likelihood 1
#@record_fn es_posterior 1
#@record_fn kls_posterior_prior 1
#@record_fn kls_posterior_likelihood 1

#record_prediction_mean!(recorder::Recorder, pred_y::AbstractVector, t::Int, k::Int) = recorder.pred_ys[:, t, k] .= pred_y
#record_prediction_covariance!(recorder::Recorder, pred_Σ::AbstractMatrix, t::Int, k::Int) = recorder.pred_Σs[:, :, t, k] .= pred_Σ

end # module
