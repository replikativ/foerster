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
├── program.cljc       # scoped programmable populations; direct PMeasure results
├── weights.cljc       # normalized selection weights and total mass
├── population/path.cljc # shared numeric/trace weighting, selection, retention
├── transition.cljc    # prepared Markov interpreter (chains and population moves)
├── transition_context.cljc # target, replay and retention context
├── transport.cljc     # scored forward/backward proposal corrections
├── particle_mcmc.cljc # common PIMH / conditional-sweep chain orchestration
├── scope.cljc cursor.cljc trace_owner.cljc # lifetime, linear handles, reclamation
├── execution.cljc     # independent CPS batch execution
├── smc.cljc smc2.cljc # savepoint/streaming/conditional and nested filters
├── trace.cljc         # policies, shared re-execution, keyed MH / finite Gibbs
├── effects.cljc       # sample / observe / factor sites
├── measure.cljc       # EmpiricalMeasure, Sample, world-descriptors
├── kernel.cljc        # kernels (prior, MH descriptions, PInferenceKernel / PMarkovKernel)
├── block.cljc hmc.cljc          # numerical blocks, HMC-within-Gibbs
├── dist.cljc          # portable distributions (raster's names/params)
├── random.cljc        # xoshiro128** streams: site-keyed, seeded
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
- JVM tests: `clojure -M:test`; SBC calibration: `clojure -M:test:sbc`;
  posteriordb accuracy and speed: `clojure -M:test:posteriordb` (results in
  `experiments/results/posteriordb/`, compare against the previous run).
  ClojureScript: `npx shadow-cljs release ci && node out/ci-tests.js`.
- Lint: `bin/lint` checks source and tests using the checked-in macro hooks.
  See `doc/inference-programs.org` for inference contracts and module boundaries.
- Formatting: `clojure -M:ffix` before committing (CI checks it).

Incremental execution restrictions and future reactive-inference admission
contracts are documented in `doc/incremental-inference.org`.
