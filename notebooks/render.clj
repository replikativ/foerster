(ns render
  "Render the tutorials to _site: `clojure -M:notebooks -m render`. Every
  notebook runs as it renders, so a broken example fails the build."
  (:require [scicloj.clay.v2.api :as clay]))

(def notebooks
  [["getting_started" "Getting Started" "a first model, sample and observe, SMC, the posterior"]
   ["algorithms" "Choosing an Inference Algorithm" "importance sampling, SMC, particle MCMC, MCMC, BBVI on known posteriors; guides; composed kernels"]
   ["worlds" "Models in Worlds" "canonical forks, budgets, what cannot be copied"]
   ["blocks" "Blocks and HMC" "numerical blocks, HMC-within-Gibbs, checking gradients"]
   ["streaming" "Streaming SMC" "online filtering against the Kalman filter"]
   ["programmable" "Programmable Inference" "the generative function interface, MH by selection, involutive MCMC"]
   ["steering" "Steering a Process" "tilting a process by a reward with SMC; trajectories as training data"]
   ["counterfactuals" "Interventions and Counterfactuals" "seeing, doing, and what would have been"]
   ["gallery" "A Gallery of Classic Models" "pencils, regression, seven scientists, label switching, branching, coal-mining disasters and eight schools, each against an exact answer"]
   ["pymc_gallery" "From the PyMC Gallery" "golf putting (two models, LOO) and radon (partial pooling, NUTS on 89 parameters) against exact posteriors; stochastic volatility by NUTS and a particle filter"]
   ["mmm" "Marketing Mix Modeling" "adstock and saturation by NUTS on a block, ROAS by counterfactual, a lift test, budget splits under risk — against a known truth"]])

(def guides
  [["language" "Language" "sites, addresses, and the rules of the spin macro"]
   ["algorithms" "Algorithms" "every inference entry point and its options"]
   ["posteriors" "Posteriors" "reading a measure: summaries, diagnostics, evidence"]
   ["extending" "Extending" "policies, proposals, MH moves and kernels"]
   ["worlds" "Worlds" "canonical worlds, their lifecycle, failure and recovery"]
   ["distributions" "Distributions and Reproducibility" "foerster.dist, seeds and streams"]
   ["design" "Design" "inference as handlers of spindel savepoints"]
   ["literature" "Literature" "the papers and systems foerster builds on"]])

(defn- index []
  (str "<html><head><meta charset='utf-8'><title>foerster</title><style>
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; line-height: 1.5; }
h1 { margin-bottom: 0.2em; } h2 { margin-top: 1.6em; }
ul { line-height: 2; } a { color: #1a6; }
</style></head><body>
<h1>foerster</h1>
<p>Probabilistic inference in forkable worlds, for Clojure.
Tutorials rendered from <a href='https://github.com/replikativ/foerster'>source</a>;
API reference on <a href='https://cljdoc.org/d/org.replikativ/foerster'>cljdoc</a>.</p>
<h2>Tutorials</h2><ul>"
       (apply str (for [[file title blurb] notebooks]
                    (str "<li><a href='foerster." file ".html'>" title "</a> — " blurb "</li>")))
       "</ul><h2>Guides</h2><ul>"
       (apply str (for [[file title blurb] guides]
                    (str "<li><a href='https://github.com/replikativ/foerster/blob/main/doc/" file ".md'>"
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
