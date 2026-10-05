"""Shared data and exact references for the comparison workloads.

W1: v0 ~ N(0,1), v_t ~ N(v_{t-1}, 0.1), y_t ~ N(v_t, 1), T = 300.
    Exact log evidence by the Kalman filter.
W3/W4: golf putting (Berry 1995), logistic regression a, b ~ N(0, 1).
    Exact posterior moments and log evidence on a grid.
"""
import json, math
import numpy as np

rng = np.random.default_rng(20261005)
T = 300
v = np.zeros(T); v[0] = rng.normal(0, 1)
for t in range(1, T): v[t] = v[t-1] + 0.1 * rng.normal()
y = v + rng.normal(size=T)

def kalman_logz(y, q=0.1**2, r=1.0, m0=0.0, p0=1.0):
    m, p, lz = m0, p0, 0.0
    for t, yt in enumerate(y):
        if t > 0: p = p + q
        s = p + r
        lz += -0.5 * (math.log(2 * math.pi * s) + (yt - m) ** 2 / s)
        k = p / s; m = m + k * (yt - m); p = (1 - k) * p
    return lz

golf = {"x": list(range(2, 21)),
        "n": [1443,694,455,353,272,256,240,217,200,237,202,192,174,167,201,195,191,147,152],
        "y": [1346,577,337,208,149,136,111,69,67,75,52,46,54,28,27,31,33,20,24]}

def golf_grid():
    from scipy.special import gammaln, expit
    x = np.array(golf["x"], float); n = np.array(golf["n"], float); k = np.array(golf["y"], float)
    A, B = np.meshgrid(np.linspace(1.9, 2.55, 401), np.linspace(-0.31, -0.21, 401), indexing="ij")
    da = (2.55 - 1.9) / 400; db = (0.31 - 0.21) / 400
    lp = -0.5 * (A**2 + B**2) - math.log(2 * math.pi)
    for xi, ni, ki in zip(x, n, k):
        p = expit(A + B * xi)
        lp = lp + gammaln(ni + 1) - gammaln(ki + 1) - gammaln(ni - ki + 1) + ki * np.log(p) + (ni - ki) * np.log1p(-p)
    m = lp.max(); w = np.exp(lp - m); Z = w.sum()
    ma = (w * A).sum() / Z; mb = (w * B).sum() / Z
    sa = math.sqrt((w * (A - ma) ** 2).sum() / Z); sb = math.sqrt((w * (B - mb) ** 2).sum() / Z)
    return {"a": [ma, sa], "b": [mb, sb], "log_evidence": m + math.log(Z * da * db)}

out = {"rw": {"y": y.tolist(), "q_sd": 0.1, "r_sd": 1.0, "log_evidence": kalman_logz(y)},
       "golf": dict(golf, exact=golf_grid()),
       "radon": {"exact": {"beta": [-0.6627248759901987, 0.0679167106648477],
                           "sigma_a": [0.3198835353108118, 0.044563934786136704],
                           "sigma_y": [0.7267554261345196, 0.01770454646763762]},
                 "source": "foerster notebooks/foerster/pymc_gallery.clj, 2-D grid over the scales, rest in closed form"}}
json.dump(out, open("data/workloads.json", "w"))
print(out["rw"]["log_evidence"], out["golf"]["exact"])
