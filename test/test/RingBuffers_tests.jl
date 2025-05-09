@testitem "Initialization" begin
    @testset "Dimensions" begin
        n = 5
        rb = RingBuffer(n, Int)

        @test typeof(rb) == RingBuffer{Int}
        @test rb.size == n
        @test rb.head == 1
        @test rb.full == false
        @test length(rb.buffer) == n
    end
end

@testitem "Functions" begin
    @testset "RingBuffer push!" begin
        n = 5
        rb = RingBuffer(n, Int)

        push!(rb, 1)
        @test rb.buffer[1] == 1
        @test rb.head == 2
        @test rb.full == false

        push!(rb, 2)
        push!(rb, 3)
        push!(rb, 4)
        push!(rb, 5)
        @test rb.buffer == [1, 2, 3, 4, 5]
        @test rb.head == 1
        @test rb.full == true
    end
    @testset "RingBuffer get_elements" begin
    n = 5
    rb = RingBuffer(n, Int)

    push!(rb, 1)
    push!(rb, 2)
    push!(rb, 3)

    @test get_elements(rb) == [1, 2, 3]

    push!(rb, 4)
    push!(rb, 5)
    @test get_elements(rb) == [1, 2, 3, 4, 5]

    push!(rb, 6)
    @test get_elements(rb) == [2, 3, 4, 5, 6]
    end
    @testset "RingBuffer get_elements_reverse" begin
        n = 5
        rb = RingBuffer(n, Int)

        push!(rb, 1)
        push!(rb, 2)
        push!(rb, 3)

        @test get_elements_reverse(rb) == [3, 2, 1]

        push!(rb, 4)
        push!(rb, 5)
        @test get_elements_reverse(rb) == [5, 4, 3, 2, 1]

        push!(rb, 6)
        @test get_elements_reverse(rb) == [6, 5, 4, 3, 2]
    end
    @testset "RingBuffer get_vector" begin
        n = 5
        rb = RingBuffer(n, Int)

        push!(rb, 1)
        push!(rb, 2)
        push!(rb, 3)

        @test get_vector(rb) == [1, 2, 3]

        push!(rb, 4)
        push!(rb, 5)
        @test get_vector(rb) == [1, 2, 3, 4, 5]

        push!(rb, 6)
        @test get_vector(rb) == [2, 3, 4, 5, 6]
    end
    @testset "RingBuffer get_last" begin
        n = 5
        rb = RingBuffer(n, Int)

        push!(rb, 1)
        @test get_last(rb) == 1

        push!(rb, 2)
        @test get_last(rb) == 2

        push!(rb, 3)
        @test get_last(rb) == 3

        push!(rb, 4)
        @test get_last(rb) == 4

        push!(rb, 5)
        @test get_last(rb) == 5

        push!(rb, 6)
        @test get_last(rb) == 6
    end
end
