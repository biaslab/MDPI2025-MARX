module Utils

using ..RingBuffers

using Plots
using Markdown
using PlutoUI
using SpecialFunctions
using Statistics
using FFTW
using MAT
using DSP

export ui_choices
export rectangle, calculate_cart_positions
export check_nan_or_inf, has_nan_or_inf
export polar2cart, cart2polar
export Point, get_values
export generateMultisineExcitation
export load_generated_controls
export getButtcoefs
export combinations_without_repetition
export get_memory_ticks
export mvgamma, logmvgamma
export mvdigamma, logmvdigamma
export not_implemented_error
export generate_filter_coeffs, setup_system_matrix
export initialize_buffer, compute_memory_size

rectangle(w, h, x, y) = Plots.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function ui_choices(values::Vector{Dict{Symbol,Any}})
    return PlutoUI.combine() do Child
        # Generate the UI elements based on the values parameter
        elements = [
            if value[:type] == :Slider
                # Include the default value for the slider
                md""" $(value[:label]): $(
                    Child(value[:label], Slider(value[:range], default = value[:default], show_value=true))
                )"""
            elseif value[:type] == :MultiCheckBox
                md""" $(value[:label]): $(
                    Child(value[:label], MultiCheckBox(value[:options], default=value[:default]))
                )"""
            elseif value[:type] == :Select
                md""" $(value[:label]): $(
                    Child(value[:label], Select(value[:options], default=value[:default]))
                )"""
            else
                throw(ArgumentError("Unsupported UI type: $(value[:type])"))
            end

            for value in values
        ]

        md"""
        $(elements)
        """
    end
end

function calculate_cart_positions(positions::Vector{Float64}, p_offset::Vector{Float64}, y_offset::Float64)
    p_carts = Vector{Vector{Float64}}()
    for pos in positions
        push!(p_carts, p_offset + [pos, y_offset])
    end
    return p_carts
end

function check_nan_or_inf(vector, errormsg::String)
    if ! has_nan_or_inf(vector) return end
    error("The vector $vector = $errormsg contains NaN or Inf.")
end

function has_nan_or_inf(vector)
    return any(x -> isnan(x) || isinf(x), vector)
end

# x = r cos θ; y = r sin θ
function polar2cart(θ::Float64, r::Float64)
    x =  sin(θ)
    y =  -cos(θ)
    #x =  cos(θ)
    #y =  sin(θ)
    return r*[x,y]
end

# r^2 = x^2 + y^2; tan θ = y/x
function cart2polar(x::Float64, y::Float64)
    r = sqrt(x^2 + y^2)
    #θ = atan(-y, x)  # Use atan2 to get the angle in the correct quadrant
    #θ = atan(y, x)
    θ = atan(-x, y) + 1.0*π
    return (θ, r)
end

struct Point
    x :: Float64
    y :: Float64
end

function Base.:+(p1::Point, p2::Point)
    return Point(p1.x + p2.x, p1.y + p2.y)
end

function Base.:*(v::Float64, p::Point)
    return Point(v*p.x, v*p.y)
end

function get_values(ps::Vector{Point})
    xs = []
    ys = []
    for p in ps
        push!(xs, p.x)
        push!(ys, p.y)
    end
    return (xs, ys)
end

function load_generated_controls(f_name::String)
    multisin_signal = matread(f_name)
    return multisin_signal["uTrain"]
end

function generateMultisineExcitation(N::Int, P::Int, M::Int, fMin::Int, fMax::Int, fs::Int; type_signal::String="odd", nGroup::Integer=3, uStd::Float64=1.0)
    """
    generates a zero-mean random phase multisine with std = 1
    INPUT
    options.N: number of points per period
    options.P: number of periods
    options.M: number of realizations
    options.fMin: minimum excited frequency
    options.fMax: maximum escited frequency
    options.fs: sample frequency
    options.type: "full", "odd", "oddrandom"

    OPTIONAL
    options.nGroup: in case of oddrandom, 1 out of nGroup odd lines is
                     discarded. Default = 3
    options.std: std of the generated signals. Default = 1

    OUTPUT
    u: NPxM record of the generated signals
    lines: excited frequency lines -> 1 = dc, 2 = fs/N

    copyright:
    Maarten Schoukens
    Vrije Universiteit Brussel, Brussels Belgium
    10/05/2017

    translated from Matlab to Julia by
    Wouter Kouw
    TU Eindhoven, Eindhoven, Netherlands
    22/01/2021

    This work is licensed under a
    Creative Commons Attribution-NonCommercial 4.0 International License
    (CC BY-NC 4.0)
    https://creativecommons.org/licenses/by-nc/4.0/
    """

    # Lines selection - select which frequencies to excite
    f0 = fs/N
    linesMin = Int64(ceil(fMin / f0) + 1)
    linesMax = Int64(floor(fMax / f0) + 1)
    lines = linesMin:linesMax

    # Remove DC component
    if lines[1] == 1; lines = lines[2:end]; end

    if type_signal == "full"
        # do nothing
    elseif type_signal == "odd"

        # remove even lines - odd indices
        if Bool(mod(lines[1],2)) # lines(1) is odd
            lines = lines[2:2:end]
        else
            lines = lines[1:2:end]
        end

    elseif type_signal == "oddrandom"

        # remove even lines - odd indices
        if Bool(mod(lines[1],2)) # lines(1) is odd
            lines = lines[2:2:end]
        else
            lines = lines[1:2:end]
        end

        # remove 1 out of nGroup lines
        nLines = length(lines)
        nRemove = floor(nLines / nGroup)
        removeInd = rand(1:nGroup, [1 nRemove])
        removeInd = removeInd + nGroup*[0:nRemove-1]
        lines = lines(!removeInd)
    end
    nLines = length(lines)

    # multisine generation - frequency domain implementation
    U = zeros(ComplexF64, N,M)

    # excite the selected frequencies
    U[lines,:] = exp.(2im*pi*rand(nLines,M))

    # go to time domain
    u = real(ifft(U))

    # rescale to obtain desired rms std
    u = uStd * u ./ std(u[:,1])

    # generate P periods
    us = repeat(u, outer=(P,1))

    return us
end

function getButtcoefs(cutoff_freq::Float64; order::Int=1, fs::Int=1)
    #dfilter = digitalfilter(DSP.Lowpass(cutoff_freq, fs=fs), DSP.Butterworth(order))
    dfilter = digitalfilter(DSP.Lowpass(cutoff_freq), DSP.Butterworth(order), fs=fs)
    ca = coefa(dfilter)
    cb = coefb(dfilter)
    return ca, cb
end

function combinations_without_repetition(A, B)
    result = []
    for i in 1:A
        for j in (i+1):B
            push!(result, (i, j))
        end
    end
    return result
end

function get_memory_ticks(memory_type::String, D_y::Int, N_y::Int, N_u::Int)
    ticks = Vector{String}()
    subscripts_digit = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    subscripts_other = ["ₜ", "₋", "₊"]
    subscript_time = "ₜ"
    subscript_time = "ₖ"
    if memory_type == "yb_ub"
        reverse_time = true
        ts = 1:N_y
        if reverse_time ts = reverse(ts) end
        for i in ts
            label = "y$(subscript_time)₋$(subscripts_digit[i])"
            for d in 1:D_y
                push!(ticks, "$(label),$(subscripts_digit[d])")
            end
        end
        ts = 1:N_u
        if reverse_time ts = reverse(ts) end
        for i in ts
            label = "u$(subscript_time)₋$(subscripts_digit[i])"
            for d in 1:D_y
                push!(ticks, "$(label),$(subscripts_digit[d])")
            end
        end
    elseif memory_type == "dim_first"
        reverse_time = false
        for d in 1:D_y
            ts = 1:N_y
            if reverse_time ts = reverse(ts) end
            for i in ts
                label = "y$(subscript_time)₋$(subscripts_digit[i])"
                push!(ticks, "$(label),$(subscripts_digit[d])")
            end
            ts = 0:N_u-1
            if reverse_time ts = reverse(ts) end
            for i in ts
                label = i == 0 ? "u$(subscript_time)" : "u$(subscript_time)₋$(subscripts_digit[i])"
                push!(ticks, "$(label),$(subscripts_digit[d])")
            end
        end
    else
        throw(ArgumentError("Unrecognized memory type: $memory_type"))
    end
    return ticks
end

function mvgamma(p::Int, a::Real)
    result = pi^(p * (p - 1) / 4)
    for j in 1:p
        result *= gamma(a + (1 - j) / 2)
    end
    return result
end

# source: StatsFuns.jl : https://github.com/JuliaStats/StatsFuns.jl/blob/master/src/misc.jl#L8-L15
function logmvgamma(p::Int, a::Real)
    # NOTE: one(a) factors are here to prevent unnecessary promotion of Float32
    res = p * (p - 1) * (log(π) * one(a)) / 4
    for ii in 1:p
        res += loggamma(a + (1 - ii) * one(a) / 2)
    end
    return res
end

# https://en.wikipedia.org/wiki/Multivariate_gamma_function
# NOTE: one(a) prevents promotion of Float32 just like StatsFun.jl (but we rarely care about that)
function mvdigamma(p::Int, a::Real)
    value = 0.0
    for i in 1:p
        value += digamma(a + 0.5*(1 - i))
    end
    return value
end

function logmvdigamma(p::Int, a::Real)
    # TODO: Is there a way to simplify a log similar to mvgamma?
    return log(mvgamma(p, a))
end

function not_implemented_error(datatype::DataType)
    func_name = stacktrace()[2].func
    error("$(func_name) not implemented for $datatype")
end

function generate_filter_coeffs(D_y::Int, N_u::Int)
    cas, cbs = [], []
    cutoff_freqs = collect((1:D_y) ./ 8)
    for i in 1:D_y
        ca, cb = getButtcoefs(cutoff_freqs[i], order=N_u - 1, fs=1)
        push!(cas, ca)
        push!(cbs, cb)
    end
    return cas, cbs
end

function setup_system_matrix(memory_type::String, D_y::Int, D_u::Int, D_x::Int, N_y::Int, N_u::Int)
    A = 1e-1 * randn(D_y, D_x)
    cas, cbs = generate_filter_coeffs(D_y, N_u)
    if memory_type == "yb_ub"
        A[1, :] = [cbs[1][2], 0.0, cbs[1][1], 0.0, cas[1][3], 0.0, cas[1][2], 0.0, cbs[1][3], 0.0]
        A[2, :] = [0.0, cbs[2][2], 0.0, cbs[2][1], 0.0, cas[2][3], 0.0, cas[2][2], 0.0, cbs[2][3]]
    elseif memory_type == "dim_first"
        for i in 1:D_y
            A[i, (i-1)*(N_y + N_u)+1 : i*(N_y + N_u)] = [cbs[i]; cas[i][2:N_u]]
        end
    else
        throw(ArgumentError("Unrecognized memory type: $memory_type"))
    end
    return A
end

function initialize_buffer(size::Int, dim::Int)
    buffer = RingBuffer(size, Vector{Float64})
    for _ in 1:size
        push!(buffer, zeros(dim))
    end
    return buffer
end

function compute_memory_size(N_y::Int, D_y::Int, N_u::Int, D_u::Int)
    return N_y * D_y + N_u * D_u
end

end # end module
