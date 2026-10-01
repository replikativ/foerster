(ns org.replikativ.foerster.kernel
  "Inference kernels: what decides a particle's latent sites.

  A `PInferenceKernel` decides a particle's latent sites during execution:
  `inference/kernel-infer` runs savepoint SMC whose sample sites take the
  value the kernel's `step` gives, which makes importance sampling vs SMC a
  choice of kernel, not of engine. The Markov-chain kernels (single-site and
  random-walk MH, block Gibbs, HMC) are descriptions that `kernel-infer` runs
  as replay plus accept over traces."
  (:refer-clojure :exclude [cycle])
  (:require [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.dist :as dist]))

;; =============================================================================
;; InferenceKernel Protocol - Checkpoint-Level Inference Control
;; =============================================================================

(defprotocol PInferenceKernel
  "Protocol for kernels that operate at the random choices of a particle.

  It decides what value a latent site of a particle takes:
  `inference/kernel-infer` asks its `step` at every sample site of savepoint
  SMC.

  Markov-chain kernels (`single-site-mh-kernel`, `random-walk-mh-kernel`,
  `block-gibbs-kernel`, `hmc-kernel`) implement `kernel-id` only: they are
  descriptions that `inference/kernel-infer` runs as replay plus accept over
  traces (`foerster.trace`)."

  (kernel-id [this]
    "Unique identifier for this kernel type (e.g., :prior, :single-site-mh).")

  (step [this ctx checkpoint trace]
    "Decide a latent site of the particle whose world is `ctx`: `checkpoint`
    is {:source distribution :options site-options :address address},
    `trace` the particle's trace so far.

    Returns {:value v} and optionally `:log-weight-delta`,
    what the value adds to the particle's weight (default 0: a draw from the
    site's distribution)."))

;; =============================================================================
;; PriorKernel - Simple Forward Sampling (Importance Sampling)
;; =============================================================================

(defrecord PriorKernel []
  PInferenceKernel

  (kernel-id [_] :prior)

  (step [_ ctx checkpoint trace]
    (let [{:keys [source options]} checkpoint
          {:keys [observe init]} options
          value (cond
                  (some? observe) observe
                  (some? init) init
                  :else (dist/draw source))]
      {:value value})))

(defn prior-kernel
  "Create a PriorKernel for simple importance sampling."
  []
  (->PriorKernel))

;; =============================================================================
;; SingleSiteMHKernel - Lightweight Metropolis-Hastings
;; =============================================================================

(defrecord SingleSiteMHKernel [num-iterations]
  PInferenceKernel
  (kernel-id [_] :single-site-mh))

(defn- chain-output
  "Attach a Markov-chain kernel's output options (extra record keys)."
  [kernel {:keys [samples burn] :or {samples :final burn 0}}]
  {:pre [(#{:final :all} samples) (nat-int? burn)]}
  (assoc kernel :samples samples :burn burn))

(defn single-site-mh-kernel
  "Create SingleSiteMHKernel for lightweight Metropolis-Hastings.

  Output options (all Markov-chain kernels): `:samples :final` (default)
  emits each chain's last state; `:samples :all` emits every state after
  the first `:burn` moves, equally weighted — an MCMC estimate from few
  chains instead of one draw per chain."
  [num-iterations & [opts]]
  {:pre [(pos-int? num-iterations)]}
  (chain-output (->SingleSiteMHKernel num-iterations) opts))

;; =============================================================================
;; RandomWalkMHKernel
;; =============================================================================

(defn- random-walk-propose
  [current-value step-size]
  (+ current-value (* step-size (dist/draw (dist/normal 0 1)))))

(defrecord RandomWalkMHKernel [num-iterations step-size]
  PInferenceKernel
  (kernel-id [_] :random-walk-mh))

(defn random-walk-mh-kernel
  "Create RandomWalkMHKernel for continuous variables.

  Metropolis-Hastings with a symmetric Gaussian proposal on one unobserved
  site per iteration; the program is replayed from that site with every other
  site held at its trace value and rescored, and the proposal is accepted on
  the ratio of joint densities (see `foerster.trace/mh-log-ratio`). A
  discrete site gets a prior proposal instead of a step.

  Output options (all Markov-chain kernels): `:samples :final` (default)
  emits each chain's last state; `:samples :all` emits every state after
  the first `:burn` moves, equally weighted — an MCMC estimate from few
  chains instead of one draw per chain."
  [num-iterations & [{:keys [step-size] :or {step-size 0.1} :as opts}]]
  {:pre [(pos-int? num-iterations) (pos? step-size)]}
  (chain-output (->RandomWalkMHKernel num-iterations step-size) opts))

;; =============================================================================
;; HMCKernel
;; =============================================================================

(defrecord HMCKernel [num-iterations step-size steps]
  PInferenceKernel
  (kernel-id [_] :hmc))

(defn hmc-kernel
  "Hamiltonian Monte Carlo on the block sites of a program
  (`foerster.block`), within Gibbs: every iteration moves each block site by
  HMC (`:step-size`, `:steps` leapfrog steps) and one other latent site by
  single-site MH (`foerster.hmc/within-gibbs`). Output options as for every
  Markov-chain kernel."
  [num-iterations & [{:keys [step-size steps] :or {step-size 0.1 steps 10} :as opts}]]
  {:pre [(pos-int? num-iterations) (pos? step-size) (pos-int? steps)]}
  (chain-output (->HMCKernel num-iterations step-size steps) opts))

;; =============================================================================
;; BlockGibbsKernel
;; =============================================================================

(defprotocol PBlockSelector
  "Protocol for selecting which block to update at each iteration."
  (select-block [this trace iteration]
    "Returns block-id (keyword) for which block to update this iteration."))

(defprotocol PBlockKernel
  "Protocol for block-level proposal kernels."
  (propose-block [this trace block-addresses]
    "Returns {addr -> proposed-value} for addresses in this block."))

(defrecord RoundRobinSelector [block-ids]
  PBlockSelector
  (select-block [_ _trace iteration]
    (nth block-ids (mod iteration (count block-ids)))))

(defn round-robin-selector
  [block-ids]
  {:pre [(seq block-ids) (every? keyword? block-ids)]}
  (->RoundRobinSelector (vec block-ids)))

(defrecord RandomSelector [block-ids]
  PBlockSelector
  (select-block [_ _trace _iteration]
    (m/pick-uniformly block-ids)))

(defn random-selector
  [block-ids]
  {:pre [(seq block-ids) (every? keyword? block-ids)]}
  (->RandomSelector (vec block-ids)))

(defrecord PriorBlockKernel []
  PBlockKernel
  (propose-block [_ trace block-addresses]
    (into {}
          (map (fn [addr]
                 (let [entry (get trace addr)
                       dist (:distribution entry)]
                   [addr (dist/draw dist)]))
               block-addresses))))

(defn prior-block-kernel [] (->PriorBlockKernel))

(defrecord RandomWalkBlockKernel [step-size]
  PBlockKernel
  (propose-block [_ trace block-addresses]
    (into {}
          (map (fn [addr]
                 (let [entry (get trace addr)
                       current (:value entry)
                       proposed (random-walk-propose current step-size)]
                   [addr proposed]))
               block-addresses))))

(defn random-walk-block-kernel
  [& [{:keys [step-size] :or {step-size 0.1}}]]
  {:pre [(pos? step-size)]}
  (->RandomWalkBlockKernel step-size))

(defrecord BlockGibbsKernel
           [num-iterations block-selector block-kernels address-classifier]
  PInferenceKernel
  (kernel-id [_] :block-gibbs))

(defn block-gibbs-kernel
  "Create a BlockGibbsKernel for block Gibbs sampling.

  Output options (all Markov-chain kernels): `:samples :final` (default)
  emits each chain's last state; `:samples :all` emits every state after
  the first `:burn` moves, equally weighted — an MCMC estimate from few
  chains instead of one draw per chain."
  [num-iterations block-selector block-kernels address-classifier & [opts]]
  {:pre [(pos-int? num-iterations)
         (satisfies? PBlockSelector block-selector)
         (map? block-kernels)
         (fn? address-classifier)]}
  (chain-output (->BlockGibbsKernel num-iterations block-selector block-kernels address-classifier)
                opts))

;; =============================================================================
;; Composing Markov-chain kernels
;; =============================================================================

(def ^:private chain-kernel-ids
  #{:single-site-mh :random-walk-mh :block-gibbs :hmc :cycle :mixture})

(defn- check-chain-kernels! [kernels]
  (when-let [bad (seq (remove #(and (satisfies? PInferenceKernel %) (chain-kernel-ids (kernel-id %))) kernels))]
    (throw (ex-info "Only Markov-chain kernels compose"
                    {:type ::not-a-chain-kernel :kernels (mapv #(when (satisfies? PInferenceKernel %) (kernel-id %)) bad)}))))

(defrecord CycleKernel [num-iterations kernels]
  PInferenceKernel
  (kernel-id [_] :cycle))

(defn cycle
  "A Markov-chain kernel that, `num-iterations` times, runs each of `kernels`
  in turn — each for its own iterations: (cycle 100 [(hmc-kernel 1 …)
  (single-site-mh-kernel 5)]) is a hundred rounds of one HMC move and five
  single-site moves. A composition of kernels that leave the posterior
  invariant leaves it invariant. Output options as for every chain kernel."
  [num-iterations kernels & [opts]]
  {:pre [(pos-int? num-iterations) (seq kernels)]}
  (check-chain-kernels! kernels)
  (chain-output (->CycleKernel num-iterations (vec kernels)) opts))

(defrecord MixtureKernel [num-iterations weights kernels]
  PInferenceKernel
  (kernel-id [_] :mixture))

(defn mixture
  "A Markov-chain kernel that, `num-iterations` times, picks one of the
  kernels in `weighted` ([[w kernel] …]) with probability ∝ w and runs it
  for its own iterations. A mixture of kernels that leave the posterior
  invariant leaves it invariant (with weights that do not depend on the
  state). Output options as for every chain kernel."
  [num-iterations weighted & [opts]]
  {:pre [(pos-int? num-iterations) (seq weighted) (every? (comp pos? first) weighted)]}
  (check-chain-kernels! (map second weighted))
  (let [total (reduce + (map first weighted))]
    (chain-output (->MixtureKernel num-iterations (mapv #(/ (first %) total) weighted) (mapv second weighted))
                  opts)))
