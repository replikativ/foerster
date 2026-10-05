include("bench_defs.jl")
mh_chain(200, 0, 99)
t0 = time_ns()
chains = [mh_chain(11000, 1000, s) for s in 1:4]
w = (time_ns() - t0) / 1e9
open("results/gen_w3.json", "w") do io; JSON.print(io, Dict("warm_s" => w, "a_draws" => [c[1] for c in chains], "b_draws" => [c[2] for c in chains])); end
println("done ", w)
