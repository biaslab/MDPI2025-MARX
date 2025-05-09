module RingBuffers

export RingBuffer, get_elements, get_elements_reverse, get_vector, get_last, length

mutable struct RingBuffer{T}
    buffer::Vector{T}   # Store the buffer
    size::Int                 # Fixed size of the buffer
    head::Int                 # Index to insert the next element
    full::Bool                # Flag to indicate if the buffer is full
end

# Constructor for the RingBuffer
function RingBuffer(n::Int, eltype::Type{T}) where T
    if eltype == Vector{Float64}
        # Initialize a buffer with empty vectors of the required size
        return RingBuffer{T}(fill([], n), n, 1, false)
    else
        # Initialize buffer with default values of type T
        return RingBuffer{T}(Vector{T}(undef, n), n, 1, false)
    end
end

# Add element `a` to the buffer with validation
function Base.push!(rb::RingBuffer{T}, a::T) where T
    rb.buffer[rb.head] = a
    rb.head += 1

    # Wrap around when we reach the end of the buffer
    if rb.head > rb.size
        rb.head = 1
        rb.full = true
    end
    return rb
end

# Get the elements of the buffer in correct order
function get_elements(rb::RingBuffer{T}) where T
    if rb.full
        return vcat(rb.buffer[rb.head:end], rb.buffer[1:rb.head-1])
    else
        return rb.buffer[1:rb.head-1]
    end
end

# Get the elements of the buffer in reverse order
function get_elements_reverse(rb::RingBuffer{T}) where T
    if rb.full
        # Return elements in reverse order, adjusting for wrap-around
        return vcat(reverse(rb.buffer[1:rb.head-1]), reverse(rb.buffer[rb.head:end]))
    else
        # If not full, just reverse the current contents
        return reverse(rb.buffer[1:rb.head-1])
    end
end

function get_vector(rb::RingBuffer{T}) where T
    return vec(hcat(get_elements(rb)...))
end

function get_last(rb::RingBuffer{T}) where T
    return rb.buffer[rb.head == 1 ? rb.size : rb.head - 1]
end

function Base.length(rb::RingBuffer{T}) where T
    return rb.full ? rb.size : rb.head - 1
end

end # module
