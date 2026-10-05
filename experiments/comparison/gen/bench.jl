# Gen.jl side of the comparison: W1 random-walk particle filter (dynamic DSL
# and static Unfold), W3 golf random-walk MH. Run from experiments/comparison:
#   julia --project=gen gen/bench.jl
# Writes results/gen.json (W3 draws are saved for ESS in Python).
using Gen, Statistics, Random
import JSON

const D = JSON.parsefile("data/workloads.json")
const ys = Float64.(D["rw"]["y"]); const T = length(ys); const exact = D["rw"]["log_evidence"]

# --- W1, dynamic DSL: re-executes the program at every step --------------
@gen function rw_dynamic(t::Int)
    v = ({(:v, 1)} ~ normal(0.0, 1.0)); {(:y, 1)} ~ normal(v, 1.0)
    for s in 2:t
        v = ({(:v, s)} ~ normal(v, 0.1)); {(:y, s)} ~ normal(v, 1.0)
    end
    v
end

function pf_dynamic(N)
    st = initialize_particle_filter(rw_dynamic, (1,), choicemap(((:y, 1), ys[1])), N)
    for t in 2:T
        maybe_resample!(st, ess_threshold=N / 2)
        particle_filter_step!(st, (t,), (UnknownChange(),), choicemap(((:y, t), ys[t])))
    end
    log_ml_estimate(st)
end

# --- W1, static Unfold: incremental, Gen's idiomatic form ----------------
@gen (static) function kernel(t::Int, prev::Float64)
    v ~ normal(t == 1 ? 0.0 : prev, t == 1 ? 1.0 : 0.1)
    y ~ normal(v, 1.0)
    return v
end
const chain = Unfold(kernel)
@gen (static) function rw_unfold(t::Int)
    vs ~ chain(t, 0.0)
    return vs
end
isdefined(Gen, Symbol("@load_generated_functions")) && @eval @load_generated_functions

function pf_unfold(N)
    st = initialize_particle_filter(rw_unfold, (1,), choicemap((:vs => 1 => :y, ys[1])), N)
    for t in 2:T
        maybe_resample!(st, ess_threshold=N / 2)
        particle_filter_step!(st, (t,), (UnknownChange(),), choicemap((:vs => t => :y, ys[t])))
    end
    log_ml_estimate(st)
end

function bench_pf(f, N, reps)
    f(N)  # warm-up / compile
    ts = Float64[]; errs = Float64[]
    for s in 1:reps
        Random.seed!(s)
        t0 = time_ns(); lz = f(N); push!(ts, (time_ns() - t0) / 1e9); push!(errs, lz - exact)
    end
    med = median(ts)
    Dict("median_s" => med, "steps_per_s" => N * T / med,
         "logZ_err_mean" => mean(errs), "logZ_err_sd" => std(errs, corrected=false))
end

# --- W3 golf random-walk MH ------------------------------------------------
const gx = Float64.(D["golf"]["x"]); const gn = Int.(D["golf"]["n"]); const gk = Int.(D["golf"]["y"])
@gen function golf(x, n)
    a ~ normal(0.0, 1.0); b ~ normal(0.0, 1.0)
    for i in 1:length(x)
        {(:y, i)} ~ binom(n[i], 1.0 / (1.0 + exp(-(a + b * x[i]))))
    end
end
@gen function walk(tr, addr, s)
    {addr} ~ normal(tr[addr], s)
end

function mh_chain(iters, burn, seed)
    Random.seed!(seed)
    # start at a = b = 0, as PyMC does (a prior draw can start where every
    # putt is certain and the likelihood is zero)
    obs = choicemap([((:y, i), gk[i]) for i in 1:length(gk)]..., (:a, 0.0), (:b, 0.0))
    tr, = generate(golf, (gx, gn), obs)
    as = Float64[]; bs = Float64[]
    for it in 1:iters
        tr, = mh(tr, walk, (:a, 0.02)); tr, = mh(tr, walk, (:b, 0.02))
        if it > burn; push!(as, tr[:a]); push!(bs, tr[:b]); end
    end
    as, bs
end

res = Dict{String,Any}()
res["W1_rw_pf_unfold"] = Dict("system" => "Gen static Unfold PF",
    "N100" => bench_pf(pf_unfold, 100, 10), "N1000" => bench_pf(pf_unfold, 1000, 5))
res["W1_rw_pf_dynamic"] = Dict("system" => "Gen dynamic-DSL PF",
    "N100" => bench_pf(pf_dynamic, 100, 3))
mh_chain(200, 0, 99)  # warm-up
t0 = time_ns()
chains = [mh_chain(11000, 1000, s) for s in 1:4]   # 2 site moves per iteration = 22000 moves, as foerster's rmh
res["W3_golf_mh"] = Dict("system" => "Gen MH (random-walk proposal per site)", "warm_s" => (time_ns() - t0) / 1e9,
    "a_draws" => [c[1] for c in chains], "b_draws" => [c[2] for c in chains])
open("results/gen.json", "w") do io; JSON.print(io, res); end
println("done")
