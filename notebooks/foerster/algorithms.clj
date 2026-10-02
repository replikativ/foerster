;; # Choosing an Inference Algorithm

;; Every algorithm in foerster runs the same kind of program: a spin with
;; `sample` and `observe` sites. What differs is what an algorithm does at
;; those sites — weigh, resample, move — and so what it is good at. This
;; notebook runs them on models whose posteriors are known exactly, shows
;; how a guide improves a proposal and how Markov-chain kernels compose, and
;; closes with advice on which to use when.

(ns foerster.algorithms
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]))

(def world (sp/create-execution-context))

(defn run
  "Run the inference `(make)` returns, seeded, and time it."
  [seed make]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        measure (sp/with-context world @(make))]
    {:measure measure
     :ms (long (/ (- (System/nanoTime) t0) 1e6))}))

(defn mean-sd
  "Posterior mean and standard deviation of `f` of the program's value."
  [measure f]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))
        xs (mapv #(double (f (m/get-value (first %)))) ps)
        mu (reduce + (map * ws xs))]
    [mu (Math/sqrt (reduce + (map (fn [w x] (* w (let [d (- x mu)] (* d d)))) ws xs)))]))

;; ## A static model
;;
;; A normal mean from two observations — the classic first benchmark of
;; [Anglican](https://github.com/probprog/anglican). The posterior is
;; N(7.25, 0.913²).

(defn gaussian []
  (spin
   (let [mu (sample (dist/normal 1.0 (Math/sqrt 5.0)) :id :mu)]
     (observe (dist/normal mu (Math/sqrt 2.0)) 9.0)
     (observe (dist/normal mu (Math/sqrt 2.0)) 8.0)
     mu)))

;; Each algorithm gets a budget of a few thousand model runs.

(def static-runs
  (array-map
   "importance sampling" #(infer/importance-sampling (gaussian) 4000)
   "SMC" #(infer/smc-infer (gaussian) 4000)
   "PIMH" #(infer/pimh-infer (gaussian) 100 40)
   "particle Gibbs" #(infer/pgibbs-infer (gaussian) 100 40)
   "PGAS" #(infer/pgas-infer (gaussian) 100 20)
   "IPMCMC" #(infer/ipmcmc-infer (gaussian) 100 10 {:num-nodes 4})
   "single-site MH" #(infer/kernel-infer (gaussian) (k/single-site-mh-kernel 2500 {:samples :all :burn 500}) 4)
   "random-walk MH" #(infer/kernel-infer (gaussian) (k/random-walk-mh-kernel 2500 {:step-size 0.5 :samples :all :burn 500}) 4)
   "BBVI" #(infer/bbvi-infer (gaussian) 200 40)))

(def seeds [1 2 3 4])

(defn summarize
  "Over `seeds`: the average and range of `(err measure)`, and the time."
  [make err]
  (let [runs (mapv #(run % make) seeds)
        errs (mapv (comp err :measure) runs)]
    {:error (/ (reduce + errs) (count errs))
     :range [(apply min errs) (apply max errs)]
     :ms (long (/ (reduce + (map :ms runs)) (count runs)))}))

;; One run of a Monte Carlo method is one draw of a random estimate, so each
;; algorithm runs with four seeds. The error is the distance of the
;; posterior mean from the truth, 7.25:

(def static-results
  (vec (for [[name make] static-runs]
         (assoc (summarize make (fn [measure] (Math/abs (- (first (mean-sd measure identity)) 7.25))))
                :algorithm name))))

(kind/table {:column-names ["algorithm" "error" "range over seeds" "ms"]
             :row-vectors (mapv (juxt :algorithm :error :range :ms) static-results)})

;; This model is harder than it looks: its prior N(1, 2.24²) puts about one
;; percent of its mass where the data point, so of the thousands of prior
;; draws that importance sampling and SMC start from, only a few dozen
;; matter. The **effective sample size** says so — it is computed from the
;; weights, (Σw)² / Σw²:

(m/effective-sample-size (:measure (run 1 #(infer/importance-sampling (gaussian) 4000))))

;; With a few dozen effective samples out of 4000, an estimate moves by
;; tenths between seeds. SMC resamples, which copies the heavy particles
;; instead of creating new values, so on a model with one static parameter
;; it cannot do better than importance sampling (and its effective sample
;; size, computed after resampling, no longer shows the problem). Particle
;; MCMC runs short SMC sweeps — each worth about one effective particle
;; here — and moves between them by an accept step, so it mixes slowly.
;; Markov chains that move from where they are, like the random walk, do
;; not depend on the prior covering the posterior; single-site MH proposes
;; from the prior, and is accepted rarely. BBVI fits a proposal to the
;; posterior and samples from that.

;; ## A sequential model
;;
;; A hidden Markov model with three states and sixteen observations. Its
;; state marginals are known exactly (by the forward–backward algorithm), so
;; we can measure each algorithm's error on all 17 × 3 of them.

(def observations [0.9 0.8 0.7 0.0 -0.025 -5.0 -2.0 -0.1 0.0 0.13 0.45 6 0.2 0.3 -1 -1])
(def transitions {0 (dist/discrete [0.1 0.5 0.4])
                  1 (dist/discrete [0.2 0.2 0.6])
                  2 (dist/discrete [0.15 0.15 0.7])})
(def emissions {0 (dist/normal -1 1) 1 (dist/normal 1 1) 2 (dist/normal 0 1)})

(defn hmm []
  (spin
   (loop [os observations
          states [(sample (dist/discrete [1.0 1.0 1.0]))]]
     (if (empty? os)
       states
       (let [s (sample (get transitions (peek states)))]
         (observe (get emissions s) (first os))
         (recur (rest os) (conj states s)))))))

(def hmm-truth
  [[0.3775 0.3092 0.3133] [0.0416 0.4045 0.5539] [0.0541 0.2552 0.6907]
   [0.0455 0.2301 0.7244] [0.1062 0.1217 0.7721] [0.0714 0.1732 0.7554]
   [0.9300 0.0001 0.0699] [0.4577 0.0452 0.4971] [0.0926 0.2169 0.6905]
   [0.1014 0.1359 0.7626] [0.0985 0.1575 0.744] [0.1781 0.2198 0.6022]
   [0.0000 0.9848 0.0152] [0.1130 0.1674 0.7195] [0.0557 0.1848 0.7595]
   [0.2017 0.0472 0.7511] [0.2545 0.0611 0.6844]])

(defn rms-error
  "Root mean square error of the posterior state marginals."
  [measure]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))
        paths (mapv (comp m/get-value first) ps)
        errs (for [t (range 17) s (range 3)]
               (- (reduce + (map (fn [w path] (if (= s (nth path t)) w 0.0)) ws paths))
                  (get-in hmm-truth [t s])))]
    (Math/sqrt (/ (reduce + (map #(* % %) errs)) (count errs)))))

(def sequential-runs
  (array-map
   "importance sampling" #(infer/importance-sampling (hmm) 4000)
   "SMC" #(infer/smc-infer (hmm) 1000)
   "particle Gibbs" #(infer/pgibbs-infer (hmm) 50 60)
   "PGAS" #(infer/pgas-infer (hmm) 50 30)
   "single-site MH" #(infer/kernel-infer (hmm) (k/single-site-mh-kernel 1000 {:samples :all :burn 200}) 4)))

(def sequential-results
  (vec (for [[name make] sequential-runs]
         (assoc (summarize make rms-error) :algorithm name))))

(kind/table {:column-names ["algorithm" "error" "range over seeds" "ms"]
             :row-vectors (mapv (juxt :algorithm :error :range :ms) sequential-results)})

;; Importance sampling proposes whole trajectories from the prior and weighs
;; them once, at the end. With sixteen observations the weight concentrates
;; on a few trajectories:

(m/effective-sample-size (:measure (run 2 #(infer/importance-sampling (hmm) 4000))))

;; SMC resamples after every observation, so the particles that explain the
;; data so far are the ones extended: with a quarter of importance
;; sampling's particles it is more than twice as accurate, in less time. Particle Gibbs and
;; PGAS iterate conditional SMC sweeps and pool them. On this short sequence
;; one SMC sweep of 1000 particles is hard to beat for the time; particle
;; MCMC pays off where one sweep degenerates — long sequences, static
;; parameters shared by all time steps — and where its MCMC guarantees
;; matter (more sweeps converge to the exact posterior).

;; ## A guide at a sample site
;;
;; Importance sampling is only as good as its proposal, and the prior is
;; often a poor one. A sample site can name its own proposal: with
;; `:proposal q`, the particle methods draw the site's value from q and weigh
;; it by p(x)/q(x), so the posterior stays the same. Take mu ~ N(0, 1) and
;; one observation 1 ~ N(mu, 0.5); the posterior is N(0.8, 0.2) and the
;; evidence N(1; 0, √1.25).

(defn guided [proposal]
  (spin
   (let [mu (if proposal
              (sample (dist/normal 0.0 1.0) :id :mu :proposal proposal)
              (sample (dist/normal 0.0 1.0) :id :mu))]
     (observe (dist/normal mu 0.5) 1.0 :id :y)
     mu)))

(def guide-runs
  (array-map
   "prior" (:measure (run 61 #(infer/importance-sampling (guided nil) 1000)))
   "guide N(0.8, 0.6)" (:measure (run 61 #(infer/importance-sampling (guided (dist/normal 0.8 0.6)) 1000)))))

(kind/table {:column-names ["proposal" "mean" "sd" "log evidence" "ESS"]
             :row-vectors (into [["exact" 0.8 (Math/sqrt 0.2) (dist/logpdf (dist/normal 0.0 (Math/sqrt 1.25)) 1.0) "–"]]
                                (for [[label measure] guide-runs]
                                  (let [[mu sd] (mean-sd measure identity)]
                                    [label mu sd (m/log-marginal measure) (m/effective-sample-size measure)])))})

;; Both estimate the same posterior and evidence; the guide, close to the
;; posterior, does it with several times the effective samples. A guide
;; fitted by BBVI, or computed from the data (amortized inference), goes
;; here. It applies to fresh draws only: replays and MH moves take
;; precedence, so a Markov chain over a guided site targets the same
;; posterior.

;; ## Composing kernels
;;
;; Markov-chain kernels compose: `k/cycle` runs each kernel in turn, and
;; `k/mixture` picks one at random by weight; each leaves the posterior
;; invariant, so the composition does. This matters when no single move
;; suits every latent. A model with a discrete and a continuous latent —
;; z ~ Bernoulli(0.3), mu ~ N(2z, 1), and 1.5 ~ N(mu, 0.5):

(defn mixed []
  (spin
   (let [z (sample (dist/bernoulli 0.3) :id :z)
         mu (sample (dist/normal (* 2.0 z) 1.0) :id :mu)]
     (observe (dist/normal mu 0.5) 1.5 :id :y)
     [z mu])))

;; Its posterior is exact: p(z | y) ∝ p(z)·N(1.5; 2z, √1.25), and given z,
;; E[mu | z, y] = (2z + 4·1.5)/5.

(def mixed-truth
  (let [lik (fn [z] (Math/exp (dist/logpdf (dist/normal (* 2.0 z) (Math/sqrt 1.25)) 1.5)))
        w1 (* 0.3 (lik 1))
        w0 (* 0.7 (lik 0))
        p1 (/ w1 (+ w0 w1))]
    {:p-z p1 :mean-mu (+ (* p1 (/ 8.0 5)) (* (- 1 p1) (/ 6.0 5)))}))

;; A random walk with a small step explores mu carefully but slowly; z
;; changes (by a prior proposal) only where mu fits the other z, and mu
;; takes many small steps to get there. Within the same budget each chain
;; stays near where it started, and the four chains disagree. Cycled or
;; mixed with single-site moves, which redraw a site from its prior, the
;; chains jump between the two modes and reach the posterior of both.

(def kernel-runs
  (array-map
   "random walk" #(infer/kernel-infer (mixed) (k/random-walk-mh-kernel 1500 {:step-size 0.05 :samples :all :burn 200}) 4)
   "cycle" #(infer/kernel-infer (mixed) (k/cycle 750 [(k/random-walk-mh-kernel 2 {:step-size 0.05})
                                                      (k/single-site-mh-kernel 1)]
                                                 {:samples :all :burn 100})
                                4)
   "mixture" #(infer/kernel-infer (mixed) (k/mixture 1500 [[2 (k/random-walk-mh-kernel 1 {:step-size 0.05})]
                                                           [1 (k/single-site-mh-kernel 1)]]
                                                     {:samples :all :burn 200})
                                  4)))

(defn chain-p-z
  "P(z = 1) within each of the four chains (`:samples :all` emits them in
  order, chain after chain)."
  [measure]
  (let [zs (mapv (comp first m/get-value first) (m/get-particles measure))]
    (mapv #(/ (reduce + %) (double (count %))) (partition-all (quot (count zs) 4) zs))))

(kind/table {:column-names ["kernel" "P(z = 1)" "E[mu]" "P(z = 1) per chain" "ms"]
             :row-vectors (into [["exact" (:p-z mixed-truth) (:mean-mu mixed-truth) "–" "–"]]
                                (for [[label make] kernel-runs]
                                  (let [{:keys [measure ms]} (run 71 make)]
                                    [label (first (mean-sd measure first)) (first (mean-sd measure second))
                                     (chain-p-z measure) ms])))})

;; The budgets are comparable — 1500 moves per chain for the random walk and
;; the mixture, 2250 for the cycle — and the composed kernels are no slower
;; per move. A cycle of an HMC move with single-site moves is the
;; same pattern for models with many continuous latents and a few discrete
;; ones.

;; ## Which one to use
;;
;; - **SMC** (`smc-infer`) is the default for most models, and the only
;;   choice for data that arrives over time (`smc/stream`, see the
;;   [streaming notebook](foerster.streaming.html)). It also estimates the
;;   evidence p(data) (`m/log-marginal`), for comparing models.
;; - **Importance sampling** (`importance-sampling`) is enough when the prior
;;   already covers the posterior well — few observations, weak evidence. Its
;;   effective sample size tells you when it is not. (After resampling, and
;;   for MCMC output, the weights no longer tell: there it counts samples.)
;; - **Particle MCMC** — PIMH (`pimh-infer`), particle Gibbs
;;   (`pgibbs-infer`), PGAS (`pgas-infer`) and IPMCMC (`ipmcmc-infer`) — runs
;;   SMC repeatedly as an MCMC kernel: more accurate than one SMC sweep for
;;   the same memory, for state-space models and static parameters alike.
;;   PGAS mixes best on long sequences.
;; - **MCMC over traces** (`kernel-infer` with `single-site-mh-kernel`,
;;   `random-walk-mh-kernel` or `block-gibbs-kernel`) when a model has many
;;   latent variables and few observations each, or structure that changes
;;   between runs. Random-walk MH for continuous latents; `k/cycle` or
;;   `k/mixture` to combine it with moves for the discrete ones.
;; - **HMC** on numerical blocks (`hmc-kernel`) when many continuous latents
;;   are correlated: see [Blocks and HMC](foerster.blocks.html).
;; - **BBVI** (`bbvi-infer`) when you need a fitted approximation q — for
;;   example to reuse as a proposal — rather than samples.
;;
;; The particle methods — importance sampling, SMC, particle MCMC, BBVI and
;; kernels on SMC — also take `:world-policy :fork`, to run in canonical
;; forks of the world they are called from: [Models in
;; Worlds](foerster.worlds.html). Markov chains (MH, random-walk MH, block
;; Gibbs, HMC) run in fresh worlds only.
