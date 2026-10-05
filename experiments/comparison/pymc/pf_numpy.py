"""W1 floor: a vectorized numpy bootstrap particle filter (systematic resampling
when ESS < N/2, as foerster's SMC). Results: results/numpy.json."""
import json, time, math
import numpy as np
D = json.load(open("data/workloads.json")); y = np.array(D["rw"]["y"]); exact = D["rw"]["log_evidence"]
def pf(N, seed):
    r = np.random.default_rng(seed); v = r.normal(0, 1, N); lw = np.zeros(N); lz = 0.0
    for t, yt in enumerate(y):
        if t > 0: v = v + 0.1 * r.normal(size=N)
        lw = lw - 0.5 * (yt - v) ** 2 - 0.5 * math.log(2 * math.pi)
        m = lw.max(); w = np.exp(lw - m); W = w / w.sum()
        if 1.0 / (W ** 2).sum() < N / 2:
            lz += m + math.log(w.mean()); lw = np.zeros(N)
            u = (r.random() + np.arange(N)) / N; v = v[np.minimum(np.searchsorted(np.cumsum(W), u), N - 1)]
    m = lw.max(); lz += m + math.log(np.exp(lw - m).mean())
    return lz
res = {}
for N in [100, 1000]:
    times, errs = [], []
    for s in range(20):
        t0 = time.perf_counter(); lz = pf(N, s); times.append(time.perf_counter() - t0); errs.append(lz - exact)
    res[f"N{N}"] = {"median_s": float(np.median(times)), "steps_per_s": N * len(y) / float(np.median(times)),
                    "logZ_err_mean": float(np.mean(errs)), "logZ_err_sd": float(np.std(errs))}
json.dump({"W1_rw_pf": {"system": "numpy bootstrap PF", **res}}, open("results/numpy.json", "w"), indent=1)
print(res)
