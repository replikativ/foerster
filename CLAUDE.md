# CLAUDE.md

Contributor notes for foerster. foerster runs on
[spindel](https://github.com/replikativ/spindel); read spindel's CLAUDE.md
first — its rules on `await`/`track` in spin bodies, effects in closures,
protocol-only runtime access and `*in-trampoline*` apply here unchanged.

## Layout

```
src/org/replikativ/foerster/
├── core.cljc          # public entry: smc-infer, kernel-infer, pgibbs-infer, …
│                      #   in-canonical-worlds (:world-policy :fork)
├── smc.cljc           # savepoint SMC, streaming, particle MCMC sweeps
├── trace.cljc         # trace policies, MH over traces, legacy trace shape
├── effects.cljc       # sample / observe / factor sites
├── measure.cljc       # EmpiricalMeasure, Sample, world-descriptors
├── kernel.cljc        # kernels (prior, MH descriptions, PInferenceKernel)
├── block.cljc hmc.cljc          # numerical blocks, HMC-within-Gibbs
├── random.cljc        # site-keyed random streams
├── gfi.cljc involutive.cljc counterfactual.cljc mechanism.cljc
└── gradient.cljc reparametrize.cljc address.cljc
```

## Rules

- Site keywords (`:inference/choose`, `:inference/factor`) are data spindel's
  handlers and traces see; don't rename them.
- A canonical particle runs the whole model (`smc/start-site`): a model's
  effects may be random without a sample site.
- Records read from a forked world's state are copies (the overlay backend
  marks them with metadata): compare ids, never `identical?`.
- JVM tests: `clojure -M:test`; SBC calibration: `clojure -M:test:sbc`.
  ClojureScript: `npx shadow-cljs release ci && node out/ci-tests.js`.
- Formatting: `clojure -M:ffix` before committing (CI checks it).
