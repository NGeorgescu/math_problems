# Paley tournament S_K search.
# Finds the smallest prime n ≡ 3 (mod 4) such that every K vertices in the
# Paley tournament on n vertices have a common dominator.
#
# OEIS A362137 indexing: A(idx) = idx-th term, where
#   A(1) = 1            (trivial: 1-vertex tournament, vacuous)
#   A(idx) = find(idx-1) for idx ≥ 2  -- smallest Paley with property S_{idx-1}.
# Known: 1, 3, 7, 19, 67, 331, 1163  ↔  A(1..7).
#
#   julia paley.jl              # sweeps A(1)..A(7)
#   julia paley.jl K            # finds smallest Paley with S_K
#   julia paley.jl K lo hi B    # custom search bounds and sample budget

using Random
using Printf

const Bitset = Vector{UInt64}

@inline bs_zero(nwords::Int) = zeros(UInt64, nwords)

@inline function bs_set!(b::Bitset, i::Int)
    @inbounds b[(i >> 6) + 1] |= UInt64(1) << (i & 63)
end

@inline function bs_copy!(dst::Bitset, src::Bitset)
    @inbounds @simd for i in eachindex(dst); dst[i] = src[i]; end
end

# In-place dst &= a; returns true iff result is non-empty.
@inline function bs_and!(dst::Bitset, a::Bitset)::Bool
    any_nz = UInt64(0)
    @inbounds @simd for i in eachindex(dst)
        dst[i] &= a[i]
        any_nz |= dst[i]
    end
    any_nz != 0
end

# out = a & b; returns true iff non-empty.
@inline function bs_and_into!(out::Bitset, a::Bitset, b::Bitset)::Bool
    any_nz = UInt64(0)
    @inbounds @simd for i in eachindex(out)
        out[i] = a[i] & b[i]
        any_nz |= out[i]
    end
    any_nz != 0
end

@inline function bs_popcount(b::Bitset)::Int
    s = 0
    @inbounds @simd for i in eachindex(b); s += count_ones(b[i]); end
    s
end

# --- Paley setup --------------------------------------------------------------

struct Paley
    n::Int
    nwords::Int
    nb::Vector{Bitset}      # nb[v+1] = {x : v - x ∈ QR}  (in-neighborhood of v)
    qr::Vector{Bool}        # qr[i+1] ⇔ i is a QR
    nrs::Vector{Int}        # sorted nonresidues in 1..n-1
    qr_inv::Vector{Int}     # qr_inv[q+1] = q^{-1} mod n for q ∈ QR; 0 elsewhere
end

function Paley_make(n::Int)::Paley
    @assert n % 4 == 3
    nwords = (n + 63) >> 6
    qr = falses(n)
    @inbounds for i in 1:n-1
        qr[((i*i) % n) + 1] = true
    end
    nrs = [j for j in 1:n-1 if !qr[j+1]]

    # nb[v+1] = {x : v - x ∈ QR}.  For n ≡ 3 (mod 4), nb[0] = NR (the non-residues),
    # and nb[v] is nb[0] rotated left by v positions in the n-bit cyclic group.  We
    # build nb[0] once, then form a "doubled" buffer (nb0 ∥ nb0) and extract a
    # length-n slice for each v.  This is O(n · nwords) total instead of O(n²).
    nb0 = bs_zero(nwords)
    @inbounds for j in 1:n-1
        if !qr[j+1]; bs_set!(nb0, j); end
    end

    src_words = ((2 * n + 63) >> 6) + 1   # +1 word so the high word read is always in-bounds
    src = zeros(UInt64, src_words)
    @inbounds for i in 1:nwords
        src[i] = nb0[i]
    end
    word_off = n >> 6
    bit_off  = n & 63
    if bit_off == 0
        @inbounds for i in 1:nwords
            src[word_off + i] |= nb0[i]
        end
    else
        @inbounds for i in 1:nwords
            src[word_off + i]     |= nb0[i] << bit_off
            src[word_off + i + 1] |= nb0[i] >> (64 - bit_off)
        end
    end

    last_bits = n & 63
    last_mask = last_bits == 0 ? typemax(UInt64) : (UInt64(1) << last_bits) - 1

    nb = Vector{Bitset}(undef, n)
    @inbounds for v in 0:n-1
        bs = bs_zero(nwords)
        start = n - v
        wo = start >> 6
        bo = start & 63
        if bo == 0
            @inbounds @simd for i in 1:nwords
                bs[i] = src[wo + i]
            end
        else
            inv = 64 - bo
            @inbounds @simd for i in 1:nwords
                bs[i] = (src[wo + i] >> bo) | (src[wo + i + 1] << inv)
            end
        end
        if last_bits != 0
            bs[nwords] &= last_mask
        end
        nb[v + 1] = bs
    end

    qr_inv = zeros(Int, n)
    @inbounds for q in 1:n-1
        if qr[q + 1]; qr_inv[q + 1] = powermod(q, n - 2, n); end
    end
    Paley(n, nwords, nb, qr, nrs, qr_inv)
end

# Canonical-form prune for case A.
#
# In case A we enumerate K-subsets S = {0, 1, t_1, …, t_{K-2}} (sorted ascending).
# Each orbit under G = {x ↦ ax + b : a ∈ QR} that contains a QR-arc has multiple
# (0,1)-representations, one per QR-arc (s, s′) ∈ S × S with s′ - s ∈ QR.
# The lex-smallest such representation is the canonical one, and we want to visit
# each orbit only once.
#
# This function returns false (i.e. "this `S` is non-canonical, prune the subtree")
# iff there is an alternative QR-arc (a, b) ≠ (0, 1) in S × S whose normalisation
# `g(x) = (x - a) * (b - a)^{-1} mod n` produces a sorted list strictly smaller
# than `S` itself.  Soundness: if it fires, the canonical rep is reached by a
# different `(t_1, …, t_{K-2})` choice in the same enumeration, so we don't lose
# any orbit.
@inline function caseA_canonical(S::Vector{Int}, K::Int, P::Paley, scratch::Vector{Int})::Bool
    n = P.n; qr = P.qr; qr_inv = P.qr_inv
    @inbounds for i in 1:K
        a = S[i]
        for j in 1:K
            i == j && continue
            (i == 1 && j == 2) && continue            # skip the (0,1) self-arc
            b = S[j]
            d = mod(b - a, n)
            d == 0 && continue
            qr[d + 1] || continue
            d_inv = qr_inv[d + 1]
            @inbounds for k in 1:K
                scratch[k] = mod((S[k] - a) * d_inv, n)
            end
            sort!(@view scratch[1:K])
            less = false
            @inbounds for k in 1:K
                if scratch[k] < S[k]
                    less = true; break
                elseif scratch[k] > S[k]
                    break
                end
            end
            less && return false
        end
    end
    true
end

# --- falsifiers ---------------------------------------------------------------

function structural_witness(P::Paley, K::Int)::Union{Nothing,Vector{Int}}
    K < 2 && return nothing
    scratch = bs_zero(P.nwords)
    # Arithmetic progressions {0, d, 2d, ..., (K-1)d}
    for d in 1:P.n-1
        bs_copy!(scratch, P.nb[1])
        s = Int[0]
        bad = false
        for j in 1:K-1
            v = mod(j * d, P.n)
            push!(s, v)
            if !bs_and!(scratch, P.nb[v + 1]); bad = true; break; end
        end
        bad && return s
    end
    # Geometric progressions {0, 1, g, g², ...}
    for g in 2:P.n-1
        bs_copy!(scratch, P.nb[1])
        if !bs_and!(scratch, P.nb[2]); return [0, 1]; end
        s = [0, 1]
        x = 1
        bad = false
        for _ in 2:K-1
            x = mod(x * g, P.n); x == 0 && break
            push!(s, x)
            if !bs_and!(scratch, P.nb[x + 1]); bad = true; break; end
        end
        bad && return s
    end
    nothing
end

function random_witness(P::Paley, rng::AbstractRNG, K::Int, budget::Int)::Union{Nothing,Vector{Int}}
    K < 2 && return nothing
    pool = collect(1:P.n-1)
    scratch = bs_zero(P.nwords)
    K - 1 > length(pool) && return nothing
    for _ in 1:budget
        s = pool[randperm(rng, length(pool))[1:K-1]]
        bs_copy!(scratch, P.nb[1])
        bad = false
        for v in s
            if !bs_and!(scratch, P.nb[v + 1]); bad = true; break; end
        end
        bad && return [0; s]
    end
    nothing
end

function local_witness(P::Paley, rng::AbstractRNG, K::Int, restarts::Int, max_swaps::Int)::Union{Nothing,Vector{Int}}
    K < 2 && return nothing
    pool = collect(1:P.n-1)
    K - 1 > length(pool) && return nothing
    m = bs_zero(P.nwords); trial = bs_zero(P.nwords); without = bs_zero(P.nwords)
    for _ in 1:restarts
        s = pool[randperm(rng, length(pool))[1:K-1]]
        bs_copy!(m, P.nb[1])
        good = true
        for v in s
            if !bs_and!(m, P.nb[v + 1]); good = false; break; end
        end
        good || return [0; s]
        for _ in 1:max_swaps
            i = rand(rng, 1:K-1)
            bs_copy!(without, P.nb[1])
            for (j, v) in enumerate(s)
                j == i && continue
                bs_and!(without, P.nb[v + 1])
            end
            best_v = s[i]; best_pop = typemax(Int)
            for _ in 1:16
                c = pool[rand(rng, 1:length(pool))]
                c in s && continue
                bs_and_into!(trial, without, P.nb[c + 1])
                p = bs_popcount(trial)
                if p < best_pop; best_pop = p; best_v = c; end
            end
            if best_v != s[i]
                s[i] = best_v
                bs_copy!(m, P.nb[1])
                ok = true
                for v in s
                    if !bs_and!(m, P.nb[v + 1]); ok = false; break; end
                end
                ok || return [0; s]
            end
        end
    end
    nothing
end

# --- deterministic verification ----------------------------------------------

# Enumerate (K-2)-subsets t of `pool` and check whether any leaf has
# (prefix ∩ ⋂ nb[v] for v in t) empty.  Returns the first such t (failure)
# or `nothing` (no counterexample under this orbit class).
function enumerate_pool(P::Paley, prefix::Bitset, pool::Vector{Int}, depth::Int)::Union{Nothing,Vector{Int}}
    if depth == 0
        return bs_popcount(prefix) == 0 ? Int[] : nothing
    end
    pool_len = length(pool)
    depth > pool_len && return nothing

    m_stack = [bs_zero(P.nwords) for _ in 0:depth]
    bs_copy!(m_stack[1], prefix)
    idx = ones(Int, depth)
    d = 1

    while true
        if idx[d] > pool_len - (depth - d)
            d == 1 && return nothing
            d -= 1; idx[d] += 1
            continue
        end
        v = pool[idx[d]]
        any_left = bs_and_into!(m_stack[d+1], m_stack[d], P.nb[v + 1])

        if !any_left
            t = [pool[idx[i]] for i in 1:d]
            nxt = idx[d] + 1
            while length(t) < depth
                nxt > pool_len && return nothing
                push!(t, pool[nxt]); nxt += 1
            end
            return t
        end

        if d == depth; idx[d] += 1; continue; end
        d += 1; idx[d] = idx[d-1] + 1
    end
end

# Parallel: split on the first element of t.  Each top-level branch is independent.
# We use Threads.@spawn for dynamic work-stealing — early branches have larger
# inner subtrees, so static partitioning leaves threads idling at the tail.
function enumerate_top_parallel(P::Paley, prefix::Bitset, pool::Vector{Int}, depth::Int)::Union{Nothing,Vector{Int}}
    if depth == 0
        return bs_popcount(prefix) == 0 ? Int[] : nothing
    end
    pool_len = length(pool)
    depth > pool_len && return nothing
    n_top = pool_len - (depth - 1)

    tasks = Vector{Task}(undef, n_top)
    for i in 1:n_top
        tasks[i] = Threads.@spawn begin
            v = pool[i]
            new_prefix = copy(prefix)
            if !bs_and!(new_prefix, P.nb[v + 1])
                # mask collapsed at depth 1 — pad out to size `depth`
                t = [v]
                nxt = i + 1
                while length(t) < depth
                    push!(t, pool[nxt]); nxt += 1
                end
                t
            elseif depth == 1
                nothing
            else
                sub_pool = collect(@view pool[i+1:end])
                sub = enumerate_pool(P, new_prefix, sub_pool, depth - 1)
                sub === nothing ? nothing : [v; sub]
            end
        end
    end

    for t in tasks
        r = fetch(t)::Union{Nothing,Vector{Int}}
        r === nothing || return r
    end
    nothing
end

# Recursive case-A walker with canonical-form pruning.  S is the current partial
# subset {0, 1, …}; we mutate it in place (push/pop).  Returns the full failing
# subset (length K) or nothing.
function _enumerate_caseA_step(P::Paley, m::Bitset, S::Vector{Int}, K::Int,
                                pool::Vector{Int}, start_idx::Int,
                                scratch::Vector{Int})::Union{Nothing,Vector{Int}}
    if length(S) == K
        return bs_popcount(m) == 0 ? copy(S) : nothing
    end
    pool_len = length(pool)
    remaining = K - length(S)
    last_i = pool_len - remaining + 1
    @inbounds for i in start_idx:last_i
        v = pool[i]
        push!(S, v)
        if !caseA_canonical(S, length(S), P, scratch)
            pop!(S); continue
        end
        new_m = copy(m)
        any_left = bs_and!(new_m, P.nb[v + 1])
        if !any_left
            # mask collapsed mid-build — pad with the smallest still-available pool elements
            result = copy(S)
            nxt = i + 1
            while length(result) < K
                if nxt > pool_len; pop!(S); return nothing; end
                push!(result, pool[nxt]); nxt += 1
            end
            pop!(S); return result
        end
        sub = _enumerate_caseA_step(P, new_m, S, K, pool, i + 1, scratch)
        if sub !== nothing; pop!(S); return sub; end
        pop!(S)
    end
    nothing
end

# Parallel case-A: split on first element via @spawn, each branch runs the
# canonical-pruning DFS serially.  Per-task scratch buffers avoid contention.
function enumerate_caseA_parallel(P::Paley, prefix::Bitset, pool::Vector{Int}, K::Int)::Union{Nothing,Vector{Int}}
    K2 = K - 2
    if K2 == 0
        return bs_popcount(prefix) == 0 ? Int[0, 1] : nothing
    end
    pool_len = length(pool)
    K2 > pool_len && return nothing
    n_top = pool_len - (K2 - 1)

    tasks = Vector{Task}(undef, n_top)
    for i in 1:n_top
        tasks[i] = Threads.@spawn begin
            v = pool[i]
            S = [0, 1, v]
            scratch = Vector{Int}(undef, K)
            if !caseA_canonical(S, 3, P, scratch)
                nothing
            else
                new_prefix = copy(prefix)
                any_left = bs_and!(new_prefix, P.nb[v + 1])
                if !any_left
                    result = copy(S)
                    nxt = i + 1
                    while length(result) < K
                        if nxt > pool_len; nothing; end
                        push!(result, pool[nxt]); nxt += 1
                    end
                    result
                elseif K == 3
                    nothing
                else
                    _enumerate_caseA_step(P, new_prefix, S, K, pool, i + 1, scratch)
                end
            end
        end
    end

    for t in tasks
        r = fetch(t)::Union{Nothing,Vector{Int}}
        r === nothing || return r
    end
    nothing
end

function verify(P::Paley, K::Int)::Union{Nothing,Vector{Int}}
    K == 0 && return nothing
    K == 1 && return bs_popcount(P.nb[1]) == 0 ? Int[0] : nothing
    g = P.nrs[1]
    # case A: S ⊇ {0, 1}
    pa = copy(P.nb[1]); bs_and!(pa, P.nb[2])
    pool_a = collect(2:P.n-1)
    t = enumerate_top_parallel(P, pa, pool_a, K - 2)
    t === nothing || return [0; 1; t]
    # case B: S ⊇ {0, g}
    pb = copy(P.nb[1]); bs_and!(pb, P.nb[g + 1])
    pool_b = [j for j in P.nrs if j != g]
    t = enumerate_top_parallel(P, pb, pool_b, K - 2)
    t === nothing || return [0; g; t]
    nothing
end

# --- prime utilities ----------------------------------------------------------

function is_prime(n::Int)::Bool
    n < 2 && return false
    n < 4 && return true
    n % 2 == 0 && return false
    i = 3
    while i * i <= n
        n % i == 0 && return false
        i += 2
    end
    true
end

function next_prime_3mod4_at_least(n::Int)::Int
    m = max(n, 3)
    if m % 4 != 3
        m += mod(3 - m, 4)
    end
    while !is_prime(m); m += 4; end
    m
end

# --- find smallest Paley with S_K ---------------------------------------------

function find_SK(K::Int; lo::Int = 3,
                 hi::Int = max(8, K * K * 4^max(K - 1, 1)),
                 sample_budget::Int = 20_000,
                 restarts::Int = 200, max_swaps::Int = 60,
                 verbose::Bool = false,
                 bookmark::Union{Nothing,String} = nothing)
    K <= 0 && return 1
    rng = Xoshiro(0xC0FFEE)

    # If a bookmark file exists and points past `lo`, resume from it.
    if bookmark !== nothing && isfile(bookmark)
        try
            saved_text = strip(read(bookmark, String))
            if startswith(saved_text, "DONE:")
                ans = parse(Int, strip(saved_text[6:end]))
                @printf "bookmark says DONE: A=%d\n" ans
                return ans
            end
            saved = parse(Int, saved_text)
            if saved > lo
                @printf "resuming from bookmark: n=%d (lo was %d)\n" saved lo
                lo = saved
            end
        catch e
            @printf "warning: could not parse bookmark %s (%s); ignoring\n" bookmark string(e)
        end
    end

    n = next_prime_3mod4_at_least(lo)
    while n < hi
        # Persist current prime BEFORE doing work, so an interrupt is recoverable.
        if bookmark !== nothing
            try; write(bookmark, string(n)); catch; end
        end
        t0 = time()
        P = Paley_make(n)
        stage = ""
        if K >= 2 && structural_witness(P, K) !== nothing
            stage = "structural"
        elseif K >= 2 && random_witness(P, rng, K, sample_budget) !== nothing
            stage = "random"
        elseif K >= 2 && local_witness(P, rng, K, restarts, max_swaps) !== nothing
            stage = "local"
        else
            v = verify(P, K)
            if v === nothing
                verbose && (@printf "n=%6d  PASS                  %.3f s\n" n (time()-t0); flush(stdout))
                if bookmark !== nothing
                    try; write(bookmark, "DONE: $n"); catch; end
                end
                return n
            else
                stage = "verify"
            end
        end
        if verbose
            @printf "n=%6d  fail (%-10s)     %.3f s\n" n stage (time()-t0)
            flush(stdout)
        end
        n = next_prime_3mod4_at_least(n + 4)
    end
    error("no answer found in [$lo, $hi) for K=$K")
end

# OEIS-aligned wrapper:  A(idx) = idx-th term of A362137.
A_oeis(idx::Int) = idx == 1 ? 1 : find_SK(idx - 1)

# --- driver -------------------------------------------------------------------

function main()
    if length(ARGS) >= 1
        K = parse(Int, ARGS[1])
        kw = Dict{Symbol,Any}(:verbose => true)
        length(ARGS) >= 2 && (kw[:lo]            = parse(Int, ARGS[2]))
        length(ARGS) >= 3 && (kw[:hi]            = parse(Int, ARGS[3]))
        length(ARGS) >= 4 && (kw[:sample_budget] = parse(Int, ARGS[4]))
        length(ARGS) >= 5 && (kw[:bookmark]      = ARGS[5])
        @printf "find_SK(K=%d)  threads=%d  lo=%s  bookmark=%s\n" K Threads.nthreads() get(kw,:lo,"-") get(kw,:bookmark,"-")
        flush(stdout)
        t = @elapsed ans = find_SK(K; kw...)
        @printf "find_SK(%d) = %d   (%.3f s)\n" K ans t
    else
        # default: sweep A(1)..A(7)
        @printf "%-6s  %-8s  %-10s\n" "idx" "A(idx)" "time"
        for idx in 1:6
            t = @elapsed ans = A_oeis(idx)
            @printf "A(%d) =  %-7d  %.3f s\n" idx ans t
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
