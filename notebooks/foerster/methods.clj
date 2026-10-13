;; # Inference Methods
;;
;; A reference for every inference method foerster implements: what it
;; does, when to use it, its main options, what it costs, a run on a model
;; whose answer is known, and the papers it comes from. The
;; [algorithms guide](https://github.com/replikativ/foerster/blob/main/doc/algorithms.md)
;; lists every option, and [Choosing an Inference
;; Algorithm](foerster.algorithms.html) compares the core methods on harder
;; models. The runs here are small, so that the notebook renders quickly;
;; the numbers are one seed's, not benchmarks.
;;
;; Every section ends with its references; the full list, with links, is at
;; the end and in
;; [doc/literature.md](https://github.com/replikativ/foerster/blob/main/doc/literature.md).

(ns foerster.methods
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.cascade :as cascade]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.optimize :as optimize]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.scmc :as scmc]
            [org.replikativ.foerster.trace :as trace]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]))

(def world (sp/create-execution-context))

(defn run
  "Run the inference `(make)` returns, seeded, in `world`, and time it."
  [seed make]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        measure (sp/with-context world @(make))]
    {:measure measure
     :ms (long (/ (- (System/nanoTime) t0) 1e6))}))

(defn particle-row
  "Label, posterior mean and sd of `f`, the weights' ESS, the log evidence,
  and the time."
  [label f {:keys [measure ms]}]
  (let [{:keys [mean sd ess]} (d/summary measure f)]
    [label mean sd ess (m/log-marginal measure) ms]))

(defn chain-row
  "Label, posterior mean and sd of `f`, R-hat and bulk ESS over the chains,
  and the time."
  [label f {:keys [measure ms]}]
  (let [{:keys [mean sd rhat ess-bulk]} (d/summary measure f)]
    [label mean sd rhat ess-bulk ms]))

(def particle-columns ["method" "mean" "sd" "ESS" "log evidence" "ms"])
(def chain-columns ["method" "mean" "sd" "R-hat" "bulk ESS" "ms"])

;; ## The models
;;
;; Most sections use one of three models with exact answers.
;;
;; **A normal mean from ten observations**: μ ~ N(0, 1) and yᵢ ~ N(μ, 1).
;; The posterior is N(Σy/(n+1), 1/(n+1)) and the evidence is a Gaussian
;; integral. The observations are spread through the program, as in a time
;; series, so the particle methods resample between them.

(def ys [0.9 1.4 0.3 1.1 0.6 1.8 0.7 1.2 0.4 1.0])

(defn normal-mean []
  (spin
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
     (loop [i 0]
       (when (< i (count ys))
         (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
         (recur (inc i))))
     mu)))

(def exact
  (let [n (count ys) s (reduce + ys) ss (reduce + (map #(* % %) ys))]
    {:mean (/ s (inc n))
     :sd (Math/sqrt (/ 1.0 (inc n)))
     :log-evidence (- (* -0.5 n (Math/log (* 2 Math/PI)))
                      (* 0.5 (Math/log (inc n)))
                      (* 0.5 (- ss (/ (* s s) (inc n)))))}))

exact

;; **A Gaussian block**: a numerical block (a site whose value is a vector
;; θ = [a b], see [Blocks and HMC](foerster.blocks.html)) whose target is
;; e⁻⁵ times the density of N(1, 0.3²) × N(−0.5, 0.3²). Its evidence is
;; therefore exactly −5. It can draw from a reference N(0, 3²) × N(0, 3²)
;; (`:sample`, with the density of that draw, `:sample-log-density`), which
;; tempering and parallel tempering start from.

(defn lnorm [x mu s] (dist/logpdf (dist/normal mu s) x))

(def log-z -5.0)

(def reference
  {:sample (fn [_] [(* 3.0 (dist/draw (dist/normal 0.0 1.0)))
                    (* 3.0 (dist/draw (dist/normal 0.0 1.0)))])
   :sample-log-density (fn [th _] (let [[a b] (vec th)] (+ (lnorm a 0.0 3.0) (lnorm b 0.0 3.0))))})

(defn gauss-lp [[a b]] (+ log-z (lnorm a 1.0 0.3) (lnorm b -0.5 0.3)))

(def gauss-block
  (block/block {:block/id :gauss :block/target :complete-conditional
                :block/latents [{:name :a :shape []} {:name :b :shape []}]}
               (merge {:log-density (fn [th _] (gauss-lp (vec th)))
                       :value+grad (fn [th _]
                                     (let [[a b] (vec th)]
                                       [(gauss-lp [a b])
                                        (double-array [(/ (- 1.0 a) 0.09) (/ (- -0.5 b) 0.09)])]))}
                      reference)))

(defn block-model
  "A model that is one block site and nothing else."
  [blk]
  (spin (let [[a b] (sample (block/block-dist blk nil) :id :theta :init [0.0 0.0])]
          {:a a :b b})))

;; **A bimodal block**: a ~ 0.3·N(−4, 0.4²) + 0.7·N(4, 0.4²), b ~ N(0, 0.4²),
;; again with evidence −5. A Markov chain started in one mode stays there;
;; the exact answer is that 70% of the mass has a > 0. Its gradient is by
;; finite differences (`block/with-numeric-gradient`), an explicit opt-in
;; for small blocks.

(defn log-sum-exp [xs]
  (let [top (reduce max xs)]
    (+ top (Math/log (reduce + (map #(Math/exp (- % top)) xs))))))

(def bimodal-block
  (block/block {:block/id :bimodal :block/target :complete-conditional
                :block/latents [{:name :a :shape []} {:name :b :shape []}]}
               (block/with-numeric-gradient
                 (merge {:log-density (fn [th _]
                                        (let [[a b] (vec th)]
                                          (+ log-z
                                             (log-sum-exp [(+ (Math/log 0.3) (lnorm a -4.0 0.4))
                                                           (+ (Math/log 0.7) (lnorm a 4.0 0.4))])
                                             (lnorm b 0.0 0.4))))}
                        reference))))

;; ## Exact enumeration
;;
;; **What**: `{:method :enumerate}` computes the exact posterior of a program
;; whose latent choices all have finite support. At every latent site the
;; particle's world forks once per value, weighted by its probability;
;; branches share the program up to where they part.
;;
;; **When**: small discrete models, and as the reference to check the other
;; methods against. **Cost**: the number of complete assignments
;; (`:max-branches`, default 100 000, bounds it). A site with infinite or
;; continuous support is refused.
;;
;; Pearl's alarm network: burglary and earthquake set off the alarm, which
;; John reports. P(burglary | John calls) is a sum over eight joint states:

(defn burglary []
  (spin
   (let [burglary (sample (dist/flip 0.01) :id :burglary)
         quake (sample (dist/flip 0.02) :id :quake)
         alarm (sample (dist/flip (cond (and burglary quake) 0.95 burglary 0.94 quake 0.29 :else 0.001)) :id :alarm)]
     (observe (dist/flip (if alarm 0.9 0.05)) true :id :john)
     burglary)))

(def burglary-exact
  (let [states (for [b [true false] e [true false] a [true false]]
                 (let [pa (cond (and b e) 0.95 b 0.94 e 0.29 :else 0.001)]
                   [b (* (if b 0.01 0.99) (if e 0.02 0.98) (if a pa (- 1 pa)) (if a 0.9 0.05))]))
        z (reduce + (map second states))]
    {:p (/ (reduce + (map second (filter first states))) z) :log-z (Math/log z)}))

(let [{:keys [measure]} (run 1 #(infer/infer (burglary) {:method :enumerate}))]
  {:particles (count (m/get-particles measure))
   :p-burglary (:mean (d/summary measure #(if % 1.0 0.0)))
   :log-evidence (m/log-marginal measure)
   :exact burglary-exact})

;; Nested inference (`infer/conditional`) uses enumeration as the inner
;; method for models of other agents' reasoning.

;; ## Importance sampling and SMC
;;
;; **What**: importance sampling (`:importance`) runs N copies of the
;; program from the prior (or a site's `:proposal`) and weights each by its
;; observations. Sequential Monte Carlo (`:smc`) parks every particle at
;; each observation and, when the effective sample size falls below
;; `:resample-threshold`·N (default 0.5), resamples: heavy particles are
;; copied (their worlds forked), light ones abandoned.
;;
;; **When**: SMC is the default for programs whose observations are spread
;; through them — time series, sequences of steps — and the only choice for
;; data that arrive over time (`smc/stream`, see [Streaming
;; SMC](foerster.streaming.html)). Both estimate the evidence p(data)
;; (`m/log-marginal`), unbiasedly.
;;
;; **Options**: `:particles`; `:resampling` — `:systematic` (default),
;; `:stratified`, `:residual`, `:multinomial`; `:policy` (constraints,
;; interventions, proposals); `:world-policy :fork` to run in forks of the
;; caller's world ([Models in Worlds](foerster.worlds.html)); `:seed`.
;; **Cost**: N runs of the program; particles are spread over the
;; executor's threads, and the result for a seed does not depend on how
;; many.

(kind/table
 {:column-names particle-columns
  :row-vectors
  (into [["exact" (:mean exact) (:sd exact) "–" (:log-evidence exact) "–"]
         (particle-row "importance sampling" identity (run 2 #(infer/infer (normal-mean) {:method :importance :particles 1000})))]
        (for [scheme [:systematic :stratified :residual :multinomial]]
          (particle-row (str "SMC, " (name scheme)) identity
                        (run 3 #(infer/smc-infer (normal-mean) 1000 {:resampling scheme})))))})

;; The resampling schemes differ in the variance they add (multinomial adds
;; the most); all are unbiased. After resampling the weights are equal, so
;; SMC's ESS counts particles, many of them copies.
;;
;; *References*: Doucet & Johansen (2011); Douc, Cappé & Moulines (2005).

;; ## Arrival-batched SMC and the particle cascade
;;
;; **What**: SMC waits at every observation for its slowest particle. With
;; `:batch B`, the particles arriving at an observation are resampled in
;; batches of B in arrival order, each batch on its own. The **particle
;; cascade** (`cascade/cascade`) has no barrier at all: a particle arriving
;; at an observation compares its weight with the running mean of the
;; weights that arrived there so far, and dies, continues, or branches into
;; several children accordingly (Paige et al.'s Eq. 14).
;;
;; **When**: particles whose steps take uneven time — a model call, a tool,
;; a simulation that sometimes runs long. **Caveats**: on a multi-threaded
;; executor the arrival order, and so the result, is not reproducible from
;; the seed; the cascade's population size varies from run to run (`:cap`,
;; default 4·n, bounds it). Both keep the evidence estimate unbiased.

(kind/table
 {:column-names particle-columns
  :row-vectors
  [["exact" (:mean exact) (:sd exact) "–" (:log-evidence exact) "–"]
   (particle-row "SMC, :batch 50" identity (run 4 #(infer/smc-infer (normal-mean) 1000 {:batch 50})))
   (particle-row "particle cascade, n = 300" identity
                 (run 5 #(spin (await (cascade/cascade (normal-mean) 300)))))]})

;; *References*: Paige, Wood, Doucet & Teh (2014); Murray, Singh & Lee
;; (2021) on why a run ends by count, not by wall clock.

;; ## Resample-move, IBIS and SMCP3
;;
;; **What**: resampling copies particles, so a static parameter sampled once
;; at the start keeps fewer and fewer distinct values. **Resample-move**
;; gives every particle Metropolis–Hastings moves after each resampling: a
;; move replays the program from an **anchor** (a kept fork of the world at
;; a latent site) to where the particle waits. Applied to a static
;; parameter with data arriving one at a time, this is Chopin's **IBIS**.
;; **SMCP3** generalizes it: a forward kernel K changes any past site and a
;; backward kernel L reweights instead of accepting or rejecting.
;;
;; **Options**: `:anchors` (a set of addresses, `{:lag L}`, `:all` or a
;; predicate), `:rejuvenate {:moves :propose}`; `:smcp3 {:forward K
;; :backward L}`. **Cost**: a move replays from its anchor, O(t) for a
;; static parameter at observation t; fresh worlds only.
;;
;; The number of distinct values of μ left at the end shows the degeneracy
;; the moves repair:

(def step-law (dist/normal 0.0 0.05))

(def random-walk-kernel
  ;; K: μ' = μ + u, u ~ N(0, 0.05²); L: u' = −u under the same law
  {:forward (fn [{mu :mu} _]
              (let [u (dist/draw step-law)]
                {:updates {:mu (+ mu u)} :log-q (dist/logpdf step-law u) :reverse (- u)}))
   :backward (fn [_ u'] (dist/logpdf step-law u'))})

(def rejuvenation-runs
  (array-map
   "SMC" (run 6 #(infer/smc-infer (normal-mean) 500))
   "resample-move" (run 6 #(infer/smc-infer (normal-mean) 500
                                            {:anchors #{:mu}
                                             :rejuvenate {:moves 2 :propose (trace/random-walk-proposal 0.2)}}))
   "SMCP3" (run 6 #(infer/smc-infer (normal-mean) 500 {:anchors #{:mu} :smcp3 random-walk-kernel}))))

(kind/table
 {:column-names (conj particle-columns "distinct μ")
  :row-vectors (into [["exact" (:mean exact) (:sd exact) "–" (:log-evidence exact) "–" "–"]]
                     (for [[label r] rejuvenation-runs]
                       (conj (particle-row label identity r) (d/distinct-count (:measure r) :mu))))})

;; In particle Gibbs (below) the retained particle moves too, by an exact
;; construction: its state before each move is drawn backwards through the
;; move, which for a reversible MH kernel is the move itself. An exact
;; enumeration of all sweep paths of a small chain checks that the sweep
;; leaves the posterior invariant to rounding error.
;;
;; *References*: Gilks & Berzuini (2001); Chopin (2002); Lew et al. (2023).

;; ## Tempered SMC
;;
;; **What**: an SMC sampler over complete program runs, through the targets
;; p(x)·L(x)^β with β rising from 0 to 1. Each step picks the next β so that
;; the conditional ESS of the reweighting is `:ess-target`·N (default 0.5),
;; resamples, and moves every particle by MH at the new temperature, random
;; walks scaled by the population's spread. With `:waste-free P`, a step
;; resamples N/P particles and keeps every state of their P-step chains.
;;
;; **When**: posteriors with separated modes, which neither a Markov chain
;; nor SMC in program order crosses; and evidence estimates for models whose
;; observations all come at the end. **Cost**: every move re-runs the
;; program from the moved site, so a step costs N program runs; fresh
;; worlds only. Block sites are tempered from their draw density to their
;; target, and a model that is one block runs on θ vectors without worlds
;; (below, with parallel tempering).

(kind/table
 {:column-names particle-columns
  :row-vectors
  [["exact" (:mean exact) (:sd exact) "–" (:log-evidence exact) "–"]
   (particle-row "tempered SMC" identity (run 7 #(infer/infer (normal-mean) {:method :tempered :particles 400})))
   (particle-row "waste-free, P = 10" identity
                 (run 7 #(infer/tempered-infer (normal-mean) 400 {:waste-free 10})))]})

;; *References*: Del Moral, Doucet & Jasra (2006); Zhou, Johansen & Aston
;; (2016); Dau & Chopin (2022).

;; ## Particle MCMC
;;
;; **What**: Markov chains whose moves are whole SMC sweeps.
;;
;; - **PIMH** (`:pimh`) proposes a fresh sweep and accepts it on the ratio
;;   of evidence estimates.
;; - **Particle Gibbs** (`:pgibbs`) runs conditional SMC: one particle is
;;   held to the trajectory kept from the previous sweep. It takes
;;   `:anchors` and `:rejuvenate` (exact resample-move, above).
;; - **PGAS** (`:pgas`): the retained particle redraws its ancestor at every
;;   observation, which mixes far better on long sequences, at the cost of a
;;   replay of the retained future per particle per observation.
;; - **IPMCMC** (`:ipmcmc`): `:num-nodes` SMC nodes, some conditional,
;;   exchanged by Gibbs updates on their evidence.
;;
;; **When**: state-space models where one sweep degenerates, and for
;; asymptotically exact answers at fixed memory. **Caveat**: their pooled
;; measures are MCMC estimates: weights are normalized per sweep, and
;; `m/log-marginal` of the pool is not an evidence estimate. For the same
;; reason particle Gibbs reports a model with zero likelihood everywhere as
;; a uniform posterior instead of failing — which is why particle MCMC is
;; not yet offered in `foerster.sandbox`.

(kind/table
 {:column-names ["method" "mean" "sd" "ms"]
  :row-vectors
  (into [["exact" (:mean exact) (:sd exact) "–"]]
        (for [[label make] [["PIMH, 100 × 20" #(infer/pimh-infer (normal-mean) 100 20)]
                            ["particle Gibbs, 50 × 40" #(infer/pgibbs-infer (normal-mean) 50 40)]
                            ["particle Gibbs with moves" #(infer/pgibbs-infer (normal-mean) 50 40
                                                                              {:anchors #{:mu}
                                                                               :rejuvenate {:moves 2 :propose (trace/random-walk-proposal 0.2)}})]
                            ["PGAS, 50 × 20" #(infer/pgas-infer (normal-mean) 50 20)]
                            ["IPMCMC, 4 nodes" #(infer/ipmcmc-infer (normal-mean) 50 10 {:num-nodes 4})]]]
          (let [{:keys [measure ms]} (run 8 make)
                {:keys [mean sd]} (d/summary measure identity)]
            [label mean sd ms])))})

;; On a static parameter like this one, particle Gibbs without moves sticks
;; to the retained value; the moves let it mix.
;;
;; *References*: Andrieu, Doucet & Holenstein (2010); Lindsten, Jordan &
;; Schön (2014); Rainforth et al. (2016).

;; ## PMMH and SMC²: static parameters of a state-space model
;;
;; **What**: **PMMH** (`:pmmh`) is a Markov chain over the parameter sites
;; θ (`:params`), each proposal scored by an SMC over the other sites: the
;; SMC's evidence estimate p̂(y | θ) stands in for the likelihood, and the
;; chain is exact for any number of particles. **SMC²** (`smc2/smc2`) carries
;; θ-particles, each with an inner streaming SMC, for data that arrive over
;; time; duplicated θ-particles fork their inner populations.
;;
;; **When**: parameters of a sequential model that SMC alone would
;; degenerate on. **Cost**: `:iterations` × `:particles` program runs; give
;; the inner SMC enough particles for a log-evidence standard deviation
;; around 1, or acceptance collapses.
;;
;; The observation noise s of a Gaussian random walk, whose likelihood the
;; Kalman filter gives exactly:

(def walk-ys [0.4 -0.6 1.9 0.2 2.8 1.1 3.5 2.0])

(defn random-walk-model []
  (spin (let [s (sample (dist/uniform 0.3 2.0) :id :s)]
          (loop [t 0 x 0.0]
            (if (= t (count walk-ys))
              s
              (let [x' (sample (dist/normal x 1.0) :id [:x t])]
                (observe (dist/normal x' s) (nth walk-ys t) :id [:y t])
                (recur (inc t) x')))))))

(defn kalman-log-lik [s]
  (loop [[y & more] walk-ys pm 0.0 pv 0.0 ll 0.0]
    (if-not y
      ll
      (let [pv (+ pv 1.0) sv (+ pv (* s s)) k (/ pv sv)]
        (recur more (+ pm (* k (- y pm))) (* (- 1.0 k) pv)
               (+ ll (dist/logpdf (dist/normal pm (Math/sqrt sv)) y)))))))

(def walk-exact-mean
  (let [h 1e-3 ss (range 0.3 2.0 h)
        ls (map #(Math/exp (kalman-log-lik %)) ss)]
    (/ (reduce + (map * ss ls)) (reduce + ls))))

(let [{:keys [measure ms]} (run 9 #(infer/infer (random-walk-model)
                                                {:method :pmmh :particles 40 :iterations 150 :burn 30
                                                 :params #{:s} :scale 0.3}))
      thetas (map :s (:thetas measure))]
  {:posterior-mean-of-s (/ (reduce + thetas) (count thetas))
   :exact walk-exact-mean
   :acceptance (:acceptance measure)
   :ms ms})

;; SMC² is a CPS operation like `smc/stream`: with the observations at
;; `:stream` sites, `(smc2/smc2 (model) {:params #{:s} :n-theta 50 :n-x
;; 100})` resolves a step whose `:push` takes the next observation and whose
;; `:measure` is the posterior over θ so far.
;;
;; *References*: Andrieu, Doucet & Holenstein (2010); Chopin, Jacob &
;; Papaspiliopoulos (2013).

;; ## Metropolis–Hastings over traces
;;
;; **What**: Markov chains over complete program traces. A move picks
;; target sites, replays the program from the earliest, proposes at the
;; targets and accepts on the full trace's MH ratio, which accounts for
;; sites that appear or disappear.
;;
;; - **Single-site MH** (`:mh`) redraws one site from its prior —
;;   lightweight MH, as Anglican's `:lmh`.
;; - **Random-walk MH** (`:rmh`, `:step-size`) perturbs continuous sites; on
;;   block sites it takes several Gaussian steps on θ with per-coordinate
;;   scales adapted during `:burn`.
;; - **Block Gibbs** (`k/block-gibbs-kernel`) moves groups of sites, each
;;   with its own kernel; **`k/cycle`** and **`k/mixture`** compose kernels
;;   (see [Choosing an Inference Algorithm](foerster.algorithms.html)).
;; - **Involutive MCMC** (`involutive/step`) and **reversible jump**: an
;;   auxiliary draw and an involution with its Jacobian, which may change
;;   the set of sites (see [Programmable Inference](foerster.programmable.html)).
;;
;; **When**: models with all their observations at the end; many latents
;; with few observations each; structure that changes between runs.
;; **Options**: `:iterations` per chain, `:chains` (default 4), `:burn`,
;; `:seed`. Markov chains run in fresh worlds only. Check R-hat (below 1.01)
;; and bulk ESS for every quantity you report.

(kind/table
 {:column-names chain-columns
  :row-vectors
  [["exact" (:mean exact) (:sd exact) "–" "–" "–"]
   (chain-row "single-site MH" identity (run 10 #(infer/infer (normal-mean) {:method :mh :iterations 2000 :burn 500})))
   (chain-row "random-walk MH" identity (run 10 #(infer/infer (normal-mean) {:method :rmh :iterations 2000 :burn 500 :step-size 0.3})))]})

;; A `:seed` makes a chain reproducible without the process generator, which
;; other sessions in the same JVM may be using: under two different process
;; seeds, the same `:seed` gives the same draws.

(let [draws (fn [process-seed]
              (mapv m/get-value (m/get-contexts
                                 (:measure (run process-seed #(infer/infer (normal-mean)
                                                                           {:method :rmh :iterations 300 :chains 2
                                                                            :step-size 0.3 :seed 42}))))))]
  {:same-draws? (= (draws 11) (draws 12))})

;; *References*: Metropolis et al. (1953); Hastings (1970); Wingate,
;; Stuhlmüller & Goodman (2011); Tierney (1994); Neklyudov et al. (2020);
;; Cusumano-Towner, Lew & Mansinghka (2020); Green (1995).

;; ## HMC and NUTS on blocks
;;
;; **What**: Hamiltonian Monte Carlo moves all latents of a block together
;; along the gradient of its log density. `k/hmc-kernel` takes a fixed step
;; size and number of leapfrog steps. **NUTS** (`:nuts`) is Stan's
;; algorithm: multinomial trajectories, the generalized U-turn criterion,
;; divergences, and during `:burn` dual-averaging step-size adaptation
;; towards `:target-accept` (0.8) and windowed adaptation of a diagonal
;; metric. `:metric :variance` (default) is Stan's regularized position
;; variance; `:metric :fisher` is nutpie's √(Var θ / Var ∇log p), the
;; diagonal that minimizes the Fisher divergence.
;;
;; **When**: many correlated continuous latents. **Requirements**: the
;; block's target must be the complete conditional of its latents, and its
;; gradient hand-written, compiled by foerster-raster, or (for small blocks)
;; numeric by explicit opt-in. **Which metric**: it depends on the
;; posterior — on posteriordb, `:fisher` gave about 2.8× the ESS per second
;; on a 3020-row logistic regression, and `:variance` was 10–40% better on
;; strongly correlated regressions. Try both.

(kind/table
 {:column-names chain-columns
  :row-vectors
  [["exact" 1.0 0.3 "–" "–" "–"]
   (chain-row "HMC, 0.1 × 8 steps" :a (run 13 #(infer/kernel-infer (block-model gauss-block)
                                                                    (k/hmc-kernel 600 {:step-size 0.1 :steps 8 :samples :all :burn 100})
                                                                    2)))
   (chain-row "NUTS, variance metric" :a (run 13 #(infer/infer (block-model gauss-block)
                                                               {:method :nuts :iterations 600 :burn 300 :chains 2})))
   (chain-row "NUTS, Fisher metric" :a (run 13 #(infer/infer (block-model gauss-block)
                                                             {:method :nuts :iterations 600 :burn 300 :chains 2 :metric :fisher})))
   (chain-row "random-walk MH on the block" :a (run 13 #(infer/infer (block-model gauss-block)
                                                                     {:method :rmh :iterations 1500 :burn 500 :chains 2})))]})

;; *References*: Duane et al. (1987); Neal (2011); Hoffman & Gelman (2014);
;; Betancourt (2017); Nesterov (2009); Seyboldt, Carlson & Carpenter (2026).

;; ## Delayed acceptance
;;
;; **What**: `k/delayed-acceptance-kernel` screens random-walk proposals on a
;; block by a cheap **coarse** log density of θ — a surrogate, a coarser
;; model, a Laplace fit — and only proposals that pass are scored by a
;; replay of the program, which accepts on its own ratio corrected by the
;; coarse one. The chain targets the program's exact posterior whatever the
;; coarse density is; a poor one only costs acceptance.
;;
;; **When**: the program is expensive and a cheap approximation exists. The
;; block need not be its complete conditional: the replay scores every
;; observation. **Options**: `{address coarse}`, `:scales` (per coordinate
;; or one), `:steps`, and the chain options.
;;
;; A conjugate regression y = a + b·x + N(0, 0.5²), (a, b) ~ N(0, I). The
;; block holds only the prior; the observations are program sites. The
;; coarse density is deliberately off — shifted, and too wide in a, too
;; narrow in b — yet the chain finds the exact posterior:

(def xs [-1.0 -0.6 -0.2 0.1 0.4 0.7 1.0 1.3])
(def line-ys [-1.9 -1.1 -0.6 0.3 0.6 1.4 1.9 2.6])

(def line-exact
  (let [s2 0.25
        sxx (reduce + (map * xs xs)) sx (reduce + xs) n (count xs)
        p00 (+ 1.0 (/ n s2)) p01 (/ sx s2) p11 (+ 1.0 (/ sxx s2))
        r0 (/ (reduce + line-ys) s2) r1 (/ (reduce + (map * xs line-ys)) s2)
        det (- (* p00 p11) (* p01 p01))]
    {:a (/ (- (* p11 r0) (* p01 r1)) det) :a-sd (Math/sqrt (/ p11 det))
     :b (/ (- (* p00 r1) (* p01 r0)) det) :b-sd (Math/sqrt (/ p00 det))}))

(def prior-block
  (block/block {:block/id :line :block/target :complete-conditional
                :block/latents [{:name :a :shape []} {:name :b :shape []}]}
               {:log-density (fn [th _] (let [[a b] (vec th)] (* -0.5 (+ (* a a) (* b b)))))
                :value+grad (fn [th _] (let [[a b] (vec th)]
                                         [(* -0.5 (+ (* a a) (* b b))) (double-array [(- a) (- b)])]))}))

(defn line-model []
  (spin (let [[a b] (sample (block/block-dist prior-block nil) :id :theta :init [0.0 0.0])]
          (loop [i 0]
            (when (< i (count xs))
              (observe (dist/normal (+ a (* b (nth xs i))) 0.5) (nth line-ys i) :id [:y i])
              (recur (inc i))))
          {:a a :b b})))

(def poor-coarse
  (let [mu [(+ (:a line-exact) 0.3) (- (:b line-exact) 0.2)]
        sds [(* 1.5 (:a-sd line-exact)) (* 0.7 (:b-sd line-exact))]]
    (fn [th] (reduce + (map (fn [x m s] (let [z (/ (- x m) s)] (* -0.5 z z))) th mu sds)))))

(let [r (run 14 #(infer/kernel-infer (line-model)
                                     (k/delayed-acceptance-kernel 2000 {:theta poor-coarse}
                                                                  {:scales 0.15 :samples :all :burn 300})
                                     4))]
  (kind/table
   {:column-names chain-columns
    :row-vectors [["exact a" (:a line-exact) (:a-sd line-exact) "–" "–" "–"]
                  (chain-row "delayed acceptance, a" :a r)
                  ["exact b" (:b line-exact) (:b-sd line-exact) "–" "–" "–"]
                  (chain-row "delayed acceptance, b" :b r)]}))

;; *References*: Christen & Fox (2005); Lykkegaard et al. (2023), whose
;; Lemma 2.4 is the two-stage ratio used here.

;; ## Tempering a block, and non-reversible parallel tempering
;;
;; A model that is one block site with `:sample` and `:sample-log-density`
;; can be tempered on its θ vectors directly, from the draw density q to
;; the target t along q^(1−β)·t^β, without a world per particle. Two
;; samplers use that path:
;;
;; - **Tempered SMC** (`:tempered`) moves a population through it, as above.
;; - **Non-reversible parallel tempering** (`:parallel-tempering`) keeps one
;;   chain per temperature (`:chains`, default 10); every scan moves each
;;   chain (an exact draw at β = 0, random-walk MH elsewhere) and offers
;;   swaps to even pairs and odd pairs in turn, so states travel through the
;;   temperatures in one direction instead of diffusing. Over `:rounds`
;;   rounds of doubling length the schedule is retuned, spacing the
;;   temperatures evenly in the communication barrier Λ the swap rejections
;;   measure. The sample is the last round's β = 1 states, and
;;   `m/log-marginal` is the stepping-stone evidence.
;;
;; **When**: multimodal or otherwise hard posteriors of a block. About 2Λ
;; chains suffice (`:barrier` reports Λ); `:round-trips` counts states that
;; went from the reference to the posterior and back. Tempered SMC spends
;; its budget on many particles, parallel tempering on long chains, whose
;; result improves with more rounds.
;;
;; On the bimodal block, where 70% of the mass has a > 0:

(def tempering-runs
  (array-map
   "tempered SMC, 1000 particles" (run 15 #(infer/infer (block-model bimodal-block) {:method :tempered :particles 1000}))
   "parallel tempering, 12 chains, 11 rounds" (run 15 #(infer/infer (block-model bimodal-block)
                                                                     {:method :parallel-tempering :chains 12 :rounds 11}))))

(kind/table
 {:column-names ["method" "P(a > 0)" "log evidence" "ms"]
  :row-vectors (into [["exact" 0.7 log-z "–"]]
                     (for [[label {:keys [measure ms]}] tempering-runs]
                       [label (:mean (d/summary measure #(if (pos? (:a %)) 1.0 0.0)))
                        (m/log-marginal measure) ms]))})

(select-keys (:measure (get tempering-runs "parallel tempering, 12 chains, 11 rounds"))
             [:barrier :round-trips :temperatures])

;; *References*: Geyer (1991); Syed, Bouchard-Côté, Deligiannidis & Doucet
;; (2022); Xie et al. (2011) for the stepping-stone estimator.

;; ## MAP, the Laplace approximation and Pathfinder
;;
;; `foerster.optimize` works on a block site's target directly, no
;; sampling, no worlds:
;;
;; - **`map-estimate`**: the mode by L-BFGS (in the latents' own
;;   coordinates, as Stan's `optimize`; `:jacobian? true` for the mode of θ).
;; - **`laplace`**: a Gaussian in θ at the mode, the Hessian by differences
;;   of the gradient, and the Laplace evidence. Exact for a Gaussian target;
;;   check it against NUTS when the posterior may be skewed.
;; - **`pathfinder`**: L-BFGS from `:paths` jittered starts; every iterate
;;   gives a Gaussian from the optimizer's inverse-Hessian estimate; a Monte
;;   Carlo ELBO with common random numbers picks the best one per path; the
;;   paths' Gaussians propose `:draws` draws that are importance-resampled
;;   against the target with Pareto-smoothed weights. `:pareto-k` above 0.7
;;   says the Gaussians miss part of the target: use the result to start a
;;   sampler, not as the posterior.
;;
;; **When**: fast approximations, initial points and reference densities
;; (tempering, delayed acceptance). **Cost**: a few hundred gradient
;; evaluations per path.

(let [site (block/block-dist gauss-block nil)
      laplace (optimize/laplace site {:init [0.0 0.0]})
      _ (random/set-seed! 16)
      pf (optimize/pathfinder site {:paths 4 :draws 500})]
  {:exact {:mean [1.0 -0.5] :sd [0.3 0.3] :log-evidence log-z}
   :map (:latents (optimize/map-estimate site {:init [0.0 0.0]}))
   :laplace {:theta (vec (:theta laplace))
             :sd (mapv #(Math/sqrt (get-in (:covariance laplace) [% %])) [0 1])
             :log-evidence (:log-evidence laplace)}
   :pathfinder {:mean (vec (:mean pf))
                :draws (count (:draws pf))
                :pareto-k (:pareto-k pf)}})

;; *References*: Liu & Nocedal (1989); Zhang, Carpenter, Gelman & Vehtari
;; (2022); Vehtari et al. (2024) for Pareto smoothing.

;; ## Black-box variational inference
;;
;; **What**: `:bbvi` learns a mean-field q, one distribution per site, by
;; stochastic gradient ascent on the ELBO with score-function gradients and
;; control variates, then returns importance samples from q;
;; `(infer/get-variational-dists measure)` returns q itself.
;;
;; **When**: you want a fitted approximation — for instance as a site's
;; `:proposal` — rather than samples. **Options**: `:particles` per
;; iteration, `:iterations`, `:base-lr`, `:robbins-monro`, `:adagrad`.
;; **Caveat**: each site's q starts at its prior and moves only by gradient.

(kind/table
 {:column-names particle-columns
  :row-vectors [["exact" (:mean exact) (:sd exact) "–" (:log-evidence exact) "–"]
                (particle-row "BBVI, 200 × 40" identity (run 17 #(infer/bbvi-infer (normal-mean) 200 40)))]})

;; Related: `learn/maximize-evidence` fits a model's parameters by maximum
;; marginal likelihood, Adam steps on central differences of an SMC
;; evidence estimate with common random numbers.
;;
;; *References*: Ranganath, Gerrish & Blei (2014); Kingma & Ba (2015).

;; ## Sequential core-set Monte Carlo and GIGA
;;
;; **What**: `scmc/scmc` is SMC over data that arrive in batches, for a model
;; given as a prior and a per-datum log-likelihood on θ vectors.
;; Resample-move over all past data costs O(t) per step, O(T²) in total;
;; SCMC moves the particles against a **core-set** of at most `:memory`
;; weighted past points whose weighted log-likelihood approximates that of
;; all the data so far. After each step the core-set plus the new batch is
;; recompressed by **GIGA** (greedy iterative geodesic ascent), fitted where
;; the particles are. Each batch reweights the particles exactly; only the
;; moves use the core-set, so the result carries its approximation error,
;; small when the log-likelihoods compress well.
;;
;; **When**: long streams of exchangeable data in bounded memory. With
;; `:memory` at least the amount of data, SCMC is resample-move SMC.

(def regression-data
  (let [rng (java.util.Random. 2)]
    (vec (repeatedly 1000 (fn [] (let [x (.nextGaussian rng)]
                                   [x (+ 0.7 (* -1.3 x) (* 0.5 (.nextGaussian rng)))]))))))

(def regression-exact
  (let [s2 0.25 n (count regression-data)
        sx (reduce + (map first regression-data)) sxx (reduce + (map (fn [[x]] (* x x)) regression-data))
        sy (reduce + (map second regression-data)) sxy (reduce + (map (fn [[x y]] (* x y)) regression-data))
        p00 (+ 1.0 (/ n s2)) p01 (/ sx s2) p11 (+ 1.0 (/ sxx s2))
        det (- (* p00 p11) (* p01 p01)) r0 (/ sy s2) r1 (/ sxy s2)]
    {:a (/ (- (* p11 r0) (* p01 r1)) det) :b (/ (- (* p00 r1) (* p01 r0)) det)
     :a-sd (Math/sqrt (/ p11 det)) :b-sd (Math/sqrt (/ p00 det))}))

(let [t0 (System/nanoTime)
      r (scmc/scmc {:prior-sample (fn [] [(dist/draw (dist/normal 0.0 1.0)) (dist/draw (dist/normal 0.0 1.0))])
                    :prior-log-density (fn [[a b]] (- (* -0.5 (+ (* a a) (* b b))) (* 2.0 0.9189385332046727)))
                    :log-lik (fn [[a b] [x y]] (let [r (- y a (* b x))]
                                                 (- (/ (* -0.5 r r) 0.25) (* 0.5 (Math/log 0.25)) 0.9189385332046727)))
                    :batches (partition 20 regression-data)}
                   {:particles 300 :memory 30 :seed [1]})
      moments (fn [j] (let [v (map #(nth % j) (:particles r)) n (count v) mu (/ (reduce + v) n)]
                        [mu (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) v)) (dec n)))]))]
  {:exact regression-exact
   :a (moments 0) :b (moments 1)
   :coreset-size (count (:coreset r)) :steps (:steps r)
   :ms (long (/ (- (System/nanoTime) t0) 1e6))})

;; `scmc/giga` is available on its own: nonnegative, at most m-sparse weights
;; over vectors whose combination approximates a target. Here it recovers
;; 3·e₀ + 2·e₂ from four orthogonal vectors:

(scmc/giga [(double-array [2 0 0 0]) (double-array [0 1 0 0]) (double-array [0 0 5 0]) (double-array [0 0 0 3])]
           (double-array [6 0 10 0])
           4)

;; *References*: Beronov, Weilbach, Wood & Campbell (2021); Campbell &
;; Broderick (2018, 2019).

;; ## Conjugate parameters, never sampled
;;
;; `foerster.conjugate` carries a conjugate parameter's posterior through
;; the program instead of a sampled value (`normal-mean`, `beta-bernoulli`,
;; `gamma-poisson`, `dirichlet-discrete`): each datum is scored under the
;; posterior predictive and the posterior updated in closed form. Under SMC
;; such a parameter never degenerates. This is the conjugate case of delayed
;; sampling (Murray et al. 2018); see the
;; [algorithms guide](https://github.com/replikativ/foerster/blob/main/doc/algorithms.md#conjugate-parameters-never-sampled).

;; ## Steering, guides, counterfactuals
;;
;; Three further uses of the same machinery have notebooks of their own:
;;
;; - **Steering a process** (`foerster.steer`): SMC over the scored steps of
;;   a process without a density — a language model, a simulator — with a
;;   value estimate as the twist (twisted SMC); `foerster.learn` turns the
;;   weighted trajectories into training data. See [Steering a
;;   Process](foerster.steering.html). *References*: Whiteley & Lee (2014);
;;   Zhao et al. (2024); Lew et al. (2023, SMC steering); Korbak et al.
;;   (2022).
;; - **Guides**: a sample site's `:proposal` draws from a distribution the
;;   program computes and weighs by log p − log q, for every particle
;;   method — see [Choosing an Inference Algorithm](foerster.algorithms.html).
;; - **Interventions and counterfactuals** (`foerster.counterfactual`):
;;   Pearl's `do` and the abduction–action–prediction counterfactual in
;;   twin worlds — see [Interventions and
;;   Counterfactuals](foerster.counterfactuals.html). *Reference*: Pearl
;;   (2009).

;; ## Where each method runs
;;
;; | Method | `:world-policy :fork` | `foerster.sandbox` | `:seed` |
;; |---|---|---|---|
;; | enumeration | – | – | not needed (exact) |
;; | importance sampling, SMC | yes | yes | yes |
;; | arrival-batched SMC, resample-move, SMCP3 | – (fresh worlds) | – | yes |
;; | particle cascade | – | – | yes |
;; | PIMH, particle Gibbs, PGAS, IPMCMC, BBVI | yes | – | yes |
;; | PMMH, SMC² | – | – | yes |
;; | MH, random-walk MH, HMC, NUTS, delayed acceptance | – | MH, RMH, NUTS | yes |
;; | tempered SMC, parallel tempering | – | – | – (process generator) |
;; | MAP, Laplace, Pathfinder | (no worlds) | yes | via `random/set-seed!` |
;; | SCMC | (no worlds) | – | yes |
;;
;; A canonical fork costs about three times a fresh world (spindel 0.1.112).
;; The sandbox offers what its sizes can bound and what fails loudly; see
;; the [algorithms guide](https://github.com/replikativ/foerster/blob/main/doc/algorithms.md#in-an-interpreter-sandbox).

;; ## References
;;
;; **SMC and its variants**
;;
;; - A. Doucet, A. M. Johansen. *A tutorial on particle filtering and
;;   smoothing: fifteen years later.* Oxford Handbook of Nonlinear
;;   Filtering, 2011.
;; - R. Douc, O. Cappé, E. Moulines. *Comparison of resampling schemes for
;;   particle filtering.* ISPA 2005. [arXiv:cs/0507025](https://arxiv.org/abs/cs/0507025)
;; - W. R. Gilks, C. Berzuini. *Following a moving target — Monte Carlo
;;   inference for dynamic Bayesian models.* JRSS B 63(1):127–146, 2001.
;;   [doi:10.1111/1467-9868.00280](https://doi.org/10.1111/1467-9868.00280)
;; - N. Chopin. *A sequential particle filter method for static models.*
;;   Biometrika 89(3):539–552, 2002. [doi:10.1093/biomet/89.3.539](https://doi.org/10.1093/biomet/89.3.539)
;; - P. Del Moral, A. Doucet, A. Jasra. *Sequential Monte Carlo samplers.*
;;   JRSS B 68(3):411–436, 2006. [doi:10.1111/j.1467-9868.2006.00553.x](https://doi.org/10.1111/j.1467-9868.2006.00553.x)
;; - Y. Zhou, A. M. Johansen, J. A. D. Aston. *Toward automatic model
;;   comparison: an adaptive sequential Monte Carlo approach.* JCGS
;;   25(3):701–726, 2016. [doi:10.1080/10618600.2015.1060885](https://doi.org/10.1080/10618600.2015.1060885)
;; - H.-D. Dau, N. Chopin. *Waste-free sequential Monte Carlo.* JRSS B
;;   84(1):114–148, 2022. [arXiv:2011.02328](https://arxiv.org/abs/2011.02328)
;; - A. K. Lew, G. Matheos, T. Zhi-Xuan, M. Ghavamizadeh, N. Gothoskar,
;;   S. Russell, V. K. Mansinghka. *SMCP3: Sequential Monte Carlo with
;;   Probabilistic Program Proposals.* AISTATS 2023, PMLR 206:7061–7088.
;;   [proceedings.mlr.press/v206/lew23a](https://proceedings.mlr.press/v206/lew23a.html)
;; - B. Paige, F. Wood, A. Doucet, Y. W. Teh. *Asynchronous Anytime
;;   Sequential Monte Carlo.* NeurIPS 2014. [arXiv:1407.2864](https://arxiv.org/abs/1407.2864)
;; - L. M. Murray, S. S. Singh, A. Lee. *Anytime Monte Carlo.* Data-Centric
;;   Engineering 2:e27, 2021. [arXiv:1612.03319](https://arxiv.org/abs/1612.03319)
;; - L. M. Murray, D. Lundén, J. Kudlicka, D. Broman, T. B. Schön. *Delayed
;;   Sampling and Automatic Rao-Blackwellization of Probabilistic Programs.*
;;   AISTATS 2018. [arXiv:1708.07787](https://arxiv.org/abs/1708.07787)
;; - B. Beronov, C. Weilbach, F. Wood, T. Campbell. *Sequential core-set
;;   Monte Carlo.* UAI 2021, PMLR 161:2165–2175.
;;   [proceedings.mlr.press/v161/beronov21a](https://proceedings.mlr.press/v161/beronov21a.html)
;; - T. Campbell, T. Broderick. *Bayesian Coreset Construction via Greedy
;;   Iterative Geodesic Ascent.* ICML 2018. [arXiv:1802.01737](https://arxiv.org/abs/1802.01737)
;; - T. Campbell, T. Broderick. *Automated Scalable Bayesian Inference via
;;   Hilbert Coresets.* JMLR 20(15), 2019. [arXiv:1710.05053](https://arxiv.org/abs/1710.05053)
;;
;; **Particle MCMC**
;;
;; - C. Andrieu, A. Doucet, R. Holenstein. *Particle Markov chain Monte
;;   Carlo methods.* JRSS B 72(3):269–342, 2010.
;;   [doi:10.1111/j.1467-9868.2009.00736.x](https://doi.org/10.1111/j.1467-9868.2009.00736.x)
;; - F. Lindsten, M. I. Jordan, T. B. Schön. *Particle Gibbs with ancestor
;;   sampling.* JMLR 15:2145–2184, 2014. [arXiv:1401.0604](https://arxiv.org/abs/1401.0604)
;; - T. Rainforth, C. A. Naesseth, F. Lindsten, B. Paige, J.-W. van de Meent,
;;   A. Doucet, F. Wood. *Interacting Particle Markov Chain Monte Carlo.*
;;   ICML 2016. [arXiv:1602.05128](https://arxiv.org/abs/1602.05128)
;; - N. Chopin, P. E. Jacob, O. Papaspiliopoulos. *SMC²: an efficient
;;   algorithm for sequential analysis of state space models.* JRSS B
;;   75(3):397–426, 2013. [arXiv:1101.1528](https://arxiv.org/abs/1101.1528)
;;
;; **Markov chains**
;;
;; - N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller,
;;   E. Teller. *Equation of State Calculations by Fast Computing Machines.*
;;   J. Chem. Phys. 21:1087, 1953.
;; - W. K. Hastings. *Monte Carlo sampling methods using Markov chains and
;;   their applications.* Biometrika 57(1):97–109, 1970.
;;   [doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
;; - D. Wingate, A. Stuhlmüller, N. Goodman. *Lightweight implementations of
;;   probabilistic programming languages via transformational compilation.*
;;   AISTATS 2011, PMLR 15:770–778.
;; - L. Tierney. *Markov chains for exploring posterior distributions.*
;;   Annals of Statistics 22(4):1701–1728, 1994.
;;   [doi:10.1214/aos/1176325750](https://doi.org/10.1214/aos/1176325750)
;; - K. Neklyudov, M. Welling, E. Egorov, D. Vetrov. *Involutive MCMC: a
;;   Unifying Framework.* ICML 2020. [arXiv:2006.16653](https://arxiv.org/abs/2006.16653)
;; - M. Cusumano-Towner, A. K. Lew, V. Mansinghka. *Automating Involutive
;;   MCMC using Probabilistic and Differentiable Programming.* 2020.
;;   [arXiv:2007.09871](https://arxiv.org/abs/2007.09871)
;; - P. J. Green. *Reversible jump Markov chain Monte Carlo computation and
;;   Bayesian model determination.* Biometrika 82(4):711–732, 1995.
;;   [doi:10.1093/biomet/82.4.711](https://doi.org/10.1093/biomet/82.4.711)
;; - J. A. Christen, C. Fox. *Markov chain Monte Carlo using an
;;   approximation.* JCGS 14(4):795–810, 2005.
;;   [doi:10.1198/106186005X76983](https://doi.org/10.1198/106186005X76983)
;; - M. B. Lykkegaard, T. J. Dodwell, C. Fox, G. Mingas, R. Scheichl.
;;   *Multilevel Delayed Acceptance MCMC.* SIAM/ASA JUQ 11(1):1–30, 2023.
;;   [doi:10.1137/22M1476770](https://doi.org/10.1137/22M1476770)
;;
;; **Hamiltonian Monte Carlo**
;;
;; - S. Duane, A. D. Kennedy, B. J. Pendleton, D. Roweth. *Hybrid Monte
;;   Carlo.* Physics Letters B 195(2):216–222, 1987.
;;   [doi:10.1016/0370-2693(87)91197-X](https://doi.org/10.1016/0370-2693(87)91197-X)
;; - R. M. Neal. *MCMC using Hamiltonian dynamics.* Handbook of Markov Chain
;;   Monte Carlo, 2011. [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)
;; - M. D. Hoffman, A. Gelman. *The No-U-Turn Sampler: Adaptively Setting
;;   Path Lengths in Hamiltonian Monte Carlo.* JMLR 15:1593–1623, 2014.
;;   [arXiv:1111.4246](https://arxiv.org/abs/1111.4246)
;; - M. Betancourt. *A Conceptual Introduction to Hamiltonian Monte Carlo.*
;;   2017. [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)
;; - Y. Nesterov. *Primal-dual subgradient methods for convex problems.*
;;   Mathematical Programming 120:221–259, 2009.
;;   [doi:10.1007/s10107-007-0149-x](https://doi.org/10.1007/s10107-007-0149-x)
;; - A. Seyboldt, E. L. Carlson, B. Carpenter. *Preconditioning Hamiltonian
;;   Monte Carlo by minimizing Fisher Divergence.* 2026.
;;   [arXiv:2603.18845](https://arxiv.org/abs/2603.18845)
;;
;; **Tempering**
;;
;; - C. J. Geyer. *Markov chain Monte Carlo maximum likelihood.* Computing
;;   Science and Statistics: Proceedings of the 23rd Symposium on the
;;   Interface, 1991.
;; - S. Syed, A. Bouchard-Côté, G. Deligiannidis, A. Doucet. *Non-reversible
;;   parallel tempering: a scalable highly parallel MCMC scheme.* JRSS B
;;   84(2):321–350, 2022. [doi:10.1111/rssb.12464](https://doi.org/10.1111/rssb.12464)
;; - W. Xie, P. O. Lewis, Y. Fan, L. Kuo, M.-H. Chen. *Improving marginal
;;   likelihood estimation for Bayesian phylogenetic model selection.*
;;   Systematic Biology 60(2):150–160, 2011.
;;   [doi:10.1093/sysbio/syq085](https://doi.org/10.1093/sysbio/syq085)
;;
;; **Optimization and variational inference**
;;
;; - D. C. Liu, J. Nocedal. *On the limited memory BFGS method for large
;;   scale optimization.* Mathematical Programming 45:503–528, 1989.
;;   [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
;; - L. Zhang, B. Carpenter, A. Gelman, A. Vehtari. *Pathfinder: Parallel
;;   quasi-Newton variational inference.* JMLR 23(306):1–49, 2022.
;;   [arXiv:2108.03782](https://arxiv.org/abs/2108.03782)
;; - A. Vehtari, D. Simpson, A. Gelman, Y. Yao, J. Gabry. *Pareto Smoothed
;;   Importance Sampling.* JMLR 25(72), 2024. [arXiv:1507.02646](https://arxiv.org/abs/1507.02646)
;; - R. Ranganath, S. Gerrish, D. Blei. *Black Box Variational Inference.*
;;   AISTATS 2014. [arXiv:1401.0118](https://arxiv.org/abs/1401.0118)
;; - D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.*
;;   ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
;;
;; **Steering and causality**
;;
;; - N. Whiteley, A. Lee. *Twisted particle filters.* Annals of Statistics
;;   42(1):115–141, 2014. [arXiv:1210.0220](https://arxiv.org/abs/1210.0220)
;; - S. Zhao, R. Brekelmans, A. Makhzani, R. Grosse. *Probabilistic
;;   Inference in Language Models via Twisted Sequential Monte Carlo.* ICML
;;   2024. [arXiv:2404.17546](https://arxiv.org/abs/2404.17546)
;; - A. K. Lew, T. Zhi-Xuan, G. Grand, V. K. Mansinghka. *Sequential Monte
;;   Carlo Steering of Large Language Models using Probabilistic Programs.*
;;   2023. [arXiv:2306.03081](https://arxiv.org/abs/2306.03081)
;; - T. Korbak, E. Perez, C. L. Buckley. *RL with KL penalties is better
;;   viewed as Bayesian inference.* Findings of EMNLP 2022.
;; - J. Pearl. *Causality: Models, Reasoning, and Inference.* 2nd ed.,
;;   Cambridge University Press, 2009.
