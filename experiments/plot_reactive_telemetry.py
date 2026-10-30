"""Plot tables exported by reactive-telemetry/-main. Requires matplotlib."""
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_table(path):
    with open(path, newline="") as source:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source, delimiter="\t")]


prefix = pathlib.Path(sys.argv[1])
fits = read_table(str(prefix) + ".fits.tsv")
updates = read_table(str(prefix) + ".updates.tsv")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")

x = [row["revision"] for row in updates]
axes[0].plot(x, [row["reference_risk"] for row in updates], "o-", label="Quadrature")
axes[0].plot(x, [row["scmc_risk"] for row in updates], "x--", label="SCMC, budget 128")
axes[0].set(xticks=x, xticklabels=["60", "180", "300", "Corrected"],
            xlabel="Observation revision", ylabel="Predictive failure probability",
            title="Prediction at high load")
axes[0].legend(fontsize=8)

budgets = sorted({row["memory"] for row in fits})
for seed in sorted({row["seed"] for row in fits}):
    rows = [row for row in fits if row["seed"] == seed]
    axes[1].plot([row["memory"] for row in rows],
                 [row["mean_error_sd"] for row in rows], "o-", alpha=0.7,
                 label=f"Seed {int(seed)}")
axes[1].axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Validation tolerance")
axes[1].set(xlabel="Coreset budget", ylabel="Max parameter mean error (posterior SD)",
            title="Small coresets can bias inference")
axes[1].legend(fontsize=7)

for seed in sorted({row["seed"] for row in fits}):
    rows = [row for row in fits if row["seed"] == seed]
    axes[2].plot([row["memory"] for row in rows],
                 [row["likelihood_calls"] / 1e6 for row in rows], "o-", alpha=0.7)
axes[2].set(xlabel="Coreset budget", ylabel="Likelihood evaluations (millions)",
            title="Inference cost, 800 particles")
for ax in axes[1:]:
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, labels=[str(int(budget)) for budget in budgets])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)

svg_path = prefix.with_suffix(".svg")
fig.savefig(svg_path)
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
fig.savefig(prefix.with_suffix(".png"), dpi=160)
