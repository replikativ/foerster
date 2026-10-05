"""PyMC side of the comparison: W2 radon NUTS, W3 golf Metropolis, W4 golf SMC.
Each workload runs twice; the second (warm, compiled) run is reported, the
first's time too. Results: results/pymc.json."""
import json, time, math, warnings
import numpy as np, pymc as pm, arviz as az, pandas as pd
warnings.simplefilter("ignore")
D = json.load(open("data/workloads.json"))
golf = D["golf"]; x = np.array(golf["x"], float); n = np.array(golf["n"]); k = np.array(golf["y"])
radon = pd.read_csv("data/radon.csv")

def golf_model():
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1); b = pm.Normal("b", 0, 1)
        pm.Binomial("y", n=n, p=pm.math.invlogit(a + b * x), observed=k)
    return m

def radon_model():
    c = radon["county"].values; f = radon["floor"].values.astype(float); y = radon["log_radon"].values
    with pm.Model() as m:
        mu_a = pm.Normal("mu_a", 0, 10); sigma_a = pm.Exponential("sigma_a", 1)
        alpha = pm.Normal("alpha", mu_a, sigma_a, shape=85)
        beta = pm.Normal("beta", 0, 10); sigma_y = pm.Exponential("sigma_y", 1)
        pm.Normal("y", alpha[c] + beta * f, sigma_y, observed=y)
    return m

def summarize(idata, var):
    s = az.summary(idata, var_names=[var], kind="all")
    return {"mean": float(s["mean"].iloc[0]), "sd": float(s["sd"].iloc[0]),
            "ess_bulk": float(s["ess_bulk"].iloc[0]), "rhat": float(s["r_hat"].iloc[0])}

def twice(f):
    t0 = time.perf_counter(); f(); first = time.perf_counter() - t0
    t0 = time.perf_counter(); r = f(); second = time.perf_counter() - t0
    return r, first, second

res = {}
# W2 radon NUTS, 4 chains x (1000 tune + 1000 draws)
m = radon_model()
idata, t1, t2 = twice(lambda: pm.sample(1000, tune=1000, chains=4, cores=4, random_seed=1, progressbar=False, model=m))
res["W2_radon_nuts"] = {"system": "pymc NUTS", "first_s": t1, "warm_s": t2,
                        **{v: summarize(idata, v) for v in ["beta", "sigma_a", "sigma_y"]}}
# W3 golf Metropolis, 4 chains x (2000 tune + 20000 draws)
m = golf_model()
idata, t1, t2 = twice(lambda: pm.sample(20000, tune=2000, chains=4, cores=4, step=pm.Metropolis(model=m), random_seed=1, progressbar=False, model=m))
res["W3_golf_mh"] = {"system": "pymc Metropolis", "first_s": t1, "warm_s": t2, **{v: summarize(idata, v) for v in ["a", "b"]}}
# W4 golf SMC (tempered), 2000 particles, 4 chains
idata, t1, t2 = twice(lambda: pm.sample_smc(2000, chains=4, cores=4, random_seed=1, progressbar=False, model=m))
lme = idata.sample_stats["log_marginal_likelihood"].values
lme = [float(np.asarray(c).ravel()[-1]) for c in lme]
res["W4_golf_smc"] = {"system": "pymc sample_smc", "first_s": t1, "warm_s": t2,
                      "log_evidence_per_chain": lme, **{v: summarize(idata, v) for v in ["a", "b"]}}
json.dump(res, open("results/pymc.json", "w"), indent=1)
print(json.dumps(res, indent=1))
