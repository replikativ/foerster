(ns render
  "Render the tutorials to _site: `clojure -M:notebooks -m render`. Every
  notebook runs as it renders, so a broken example fails the build."
  (:require [scicloj.clay.v2.api :as clay]))

(def notebooks
  [["getting_started" "Getting Started" "a first model, sample and observe, SMC, the posterior"]
   ["queue" "Inference in a Running Application" "a live queue, speculative application state, SMC checked against enumeration"]
   ["algorithms" "Choosing an Inference Algorithm" "importance sampling, SMC, particle MCMC, MCMC, BBVI on known posteriors; guides; composed kernels"]
   ["methods" "Inference Methods" "every inference method with a run on a known answer, its options, its cost and its papers"]
   ["worlds" "Models in Worlds" "canonical forks, budgets, what cannot be copied"]
   ["blocks" "Blocks and HMC" "numerical blocks, HMC-within-Gibbs, checking gradients"]
   ["streaming" "Streaming SMC" "online filtering against the Kalman filter"]
   ["programmable" "Trace Programming and Custom Moves" "the generative function interface, MH by selection, involutive MCMC"]
   ["steering" "Steering a Process" "tilting a process by a reward with SMC; trajectories as training data"]
   ["counterfactuals" "Interventions and Counterfactuals" "seeing, doing, and what would have been"]
   ["causal_workflow" "From Data to Counterfactual Repairs" "fit mechanisms, check held-out data, quantify effect and attribution uncertainty, and compare repairs"]
   ["gallery" "A Gallery of Classic Models" "pencils, regression, seven scientists, label switching, branching, coal-mining disasters and eight schools, each against an exact answer"]
   ["pymc_gallery" "From the PyMC Gallery" "golf putting (two models, LOO) and radon (partial pooling, NUTS on 89 parameters) against exact posteriors; stochastic volatility by NUTS and a particle filter"]
   ["mmm" "Marketing Mix Modeling" "adstock and saturation by NUTS on a block, ROAS by counterfactual, a lift test, budget splits under risk — against a known truth"]])

(def guides
  [["concepts.md" "Concepts and Reading Paths" "worlds, traces, measures, populations and numerical blocks"]
   ["language.md" "Language" "sites, addresses and explicit spin scopes"]
   ["algorithms.md" "Algorithms" "entry points and their model requirements"]
   ["posteriors.md" "Posteriors" "summaries, diagnostics and evidence"]
   ["workflow.md" "Checking a Model and its Inference" "predictive checks, convergence diagnostics and calibration"]
   ["coming-from.md" "Coming from Other PPLs" "Stan, PyMC, Turing, Gen, Anglican and WebPPL"]
   ["inference-programs.org" "Inference Programs" "scoped populations, targets, weighting, moves and transport"]
   ["incremental-inference.org" "Incremental Trace Inference" "keyed reuse, finite Gibbs and the reactive boundary"]
   ["causal.md" "Causal Queries and Identifiability" "structural assumptions, counterfactual worlds, identification and attribution"]
   ["amortized.md" "Amortized Inference" "simulator schemas, masks, artifacts and validation limits"]
   ["extending.md" "Extending" "distributions, policies, proposals and kernels"]
   ["worlds.md" "Worlds" "canonical worlds, lifecycle, resources and recovery"]
   ["distributions.md" "Distributions and Reproducibility" "foerster.dist, seeds and streams"]
   ["design.md" "Design" "savepoints, traces and numerical engines"]
   ["literature.md" "Literature" "papers and systems foerster builds on"]])

(defn- index []
  (str "<html><head><meta charset='utf-8'><title>foerster</title><style>
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; line-height: 1.5; }
h1 { margin-bottom: 0.2em; } h2 { margin-top: 1.6em; }
ul { line-height: 2; } a { color: #1a6; }
</style></head><body>
<h1>foerster</h1>
<p>Probabilistic inference in forkable worlds, for Clojure and ClojureScript.
Tutorials rendered from <a href='https://github.com/replikativ/foerster'>source</a>;
API reference on <a href='https://cljdoc.org/d/org.replikativ/foerster'>cljdoc</a>.</p>
<h2>Tutorials</h2><ul>"
       (apply str (for [[file title blurb] notebooks]
                    (str "<li><a href='foerster." file ".html'>" title "</a> — " blurb "</li>")))
       "</ul><h2>Guides</h2><ul>"
       (apply str (for [[file title blurb] guides]
                    (str "<li><a href='https://github.com/replikativ/foerster/blob/main/doc/" file "'>"
                         title "</a> — " blurb "</li>")))
       "</ul></body></html>"))

(defn -main [& _]
  (doseq [[file] notebooks]
    (println "Rendering" file)
    (clay/make! {:source-path (str "notebooks/foerster/" file ".clj")
                 :base-target-path "_site"
                 :show false}))
  (spit "_site/index.html" (index))
  (println "Done.")
  (shutdown-agents)
  (System/exit 0))
