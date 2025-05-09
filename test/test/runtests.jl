using Aqua
using CpuId # cputhreads, cpucores
using ReTestItems # parallel testing
using MARX

# https://juliatesting.github.io/Aqua.jl/stable/test_all/
Aqua.test_all(
    MARX;
    ambiguities = false,
    unbound_args = true,
    undefined_exports = true,
    project_extras = true,
    stale_deps=false,
    deps_compat = false,
    piracies = false,
    persistent_tasks = false
)

# TODO: cputhreads/cpucores returns 0 even using env variable JULIA_NUM_THREADS='auto' or JULIA_NUM_THREADS=4
nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

# macOS: cores/threads = sysctl -n hw.physicalcpu, sysctl -n hw.logicalcpu
ncores=10
nthreads=10

# Test files in test directory with suffix _tests.jl or _test.jl
# TODO: Make sure julia starts with more workers
runtests(MARX; nworkers = ncores, nworker_threads = Int(nthreads / ncores), memory_threshold = 1.0)
