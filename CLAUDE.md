# CLAUDE.md

Contributor notes for foerster. foerster runs on
[spindel](https://github.com/replikativ/spindel); read spindel's CLAUDE.md
first — its rules on `await`/`track` in spin bodies, effects in closures,
protocol-only runtime access and `*in-trampoline*` apply here unchanged.

## Layout

```
src/org/replikativ/foerster/
├── core.cljc           # public entry: infer and every method's entry point;
│                       #   errors carry core's keywords wherever raised
├── dist.cljc effects.cljc block.cljc   # what models are written with
├── measure.cljc diagnostics.cljc random.cljc weights.cljc
│                       #   results, checks, seeding, population mass
├── program.cljc        # user inference programs over populations
├── sandbox.clj         # the interpreter-sandbox surface (dvergr)
├── analysis.clj amortized.cljc         # facades of analysis/ and amortized/
├── model/              # mechanism process conjugate: model-language extensions
├── runtime/            # scope execution cursor trace trace_owner gfi canonical:
│                       #   worlds, traces, lifetimes, the frozen caller fork
├── population/         # path (the population-path interpreter), theta
│                       #   (populations on θ vectors)
├── moves/              # kernel transition transition_context transport involutive
│                       #   block_mh hmc nuts: Markov moves and their interpreter
├── inference/          # worlds (world policy, options, canonical runs) smc smc2
│                       #   enumerate cascade particle_mcmc pmcmc tempering
│                       #   parallel_tempering scmc bbvi gradient optimize compiled
│                       #   posterior (query, predict, predictive) steer
├── diagnostics/arviz.cljc
├── causal/             # identification model counterfactual attribution
├── amortized/          # compiler learn
└── analysis/           # derive
```

The top level is the public API (what users require to write models and read
results) and its foundations; folders hold the implementation. Layers, lowest
first: `random`; `dist`, `weights`; `measure`, `effects`, `block`;
`diagnostics`; `model/`; `runtime/`; `analysis/`; `moves/`; `population/`;
`inference/`; `causal/`; `core`, `program`; `amortized/`, `sandbox`. A
namespace requires only its own layer and those below;
`test/org/replikativ/foerster/layout_test.clj` checks it, so a new namespace
needs a layer there.

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
