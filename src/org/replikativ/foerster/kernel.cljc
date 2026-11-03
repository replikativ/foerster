(ns org.replikativ.foerster.kernel
  "Inference kernels: what decides a particle's latent sites.

  A `PInferenceKernel` decides a particle's latent sites during execution:
  `inference/kernel-infer` runs savepoint SMC whose sample sites take the
  value the kernel's `step` gives, which makes importance sampling vs SMC a
  choice of proposal policy, not of engine. `PMarkovKernel` describes
  transitions that `foerster.transition` compiles and runs over traces."
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

  Site kernels choose and correct proposals; they are distinct from
  target-invariant Markov transitions described by `PMarkovKernel`."

  (kernel-id [this]
    "Identifier for this site-proposal policy (e.g., :prior).")

  (step [this ctx checkpoint trace]
    "Decide a latent site of the particle whose world is `ctx`: `checkpoint`
    is {:source distribution :options site-options :address address},
    `trace` the particle's trace so far.

    Returns {:value v} and optionally `:log-weight-delta`,
    what the value adds to the particle's weight (default 0: a draw from the
    site's distribution)."))

(defprotocol PMarkovKernel
  "Descriptions of Markov transitions, distinct from site proposal policies.
  Fixed-tuning transitions preserve their target under the individual kernel's
  assumptions; warm-up is not an evidence-estimating inference algorithm."
  (transition-id [this] "The transition family compiled by foerster.transition."))

(defn markov-kernel? [kernel]
  (satisfies? PMarkovKernel kernel))

;; =============================================================================
;; PriorKernel - Simple Forward Sampling (Importance Sampling)
;; =============================================================================

(defrecord PriorKernel []
  PInferenceKernel

  (kernel-id [_] :prior)

  (step [_ _ctx checkpoint _trace]
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
  PMarkovKernel
  (transition-id [_] :single-site-mh))

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

(defrecord FiniteGibbsKernel [num-iterations address]
  PMarkovKernel
  (transition-id [_] :finite-gibbs))

(defn finite-gibbs-kernel
  "Exact conditional draws at one fixed finite-support sample address.
  Other latent values and the scored site structure must remain unchanged
  across candidates; unsupported structure/support changes are refused.
  Keyed sites update their key; other sites replay from a valid anchor.
  Output options are :samples and :burn, as for single-site-mh-kernel.
  Compose fixed addresses with cycle/mixture for multiple-site sweeps."
  [num-iterations address & [opts]]
  {:pre [(pos-int? num-iterations) (some? address)]}
  (chain-output (->FiniteGibbsKernel num-iterations address) opts))

;; =============================================================================
;; RandomWalkMHKernel
;; =============================================================================

(defn- random-walk-propose
  [current-value step-size]
  (+ current-value (* step-size (dist/draw (dist/normal 0 1)))))

(defrecord RandomWalkMHKernel [num-iterations step-size]
  PMarkovKernel
  (transition-id [_] :random-walk-mh))

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
;; DelayedAcceptanceKernel
;; =============================================================================

(defrecord DelayedAcceptanceKernel [num-iterations coarse scales steps step-size]
  PMarkovKernel
  (transition-id [_] :delayed-acceptance))

(defn delayed-acceptance-kernel
  "Two-stage delayed acceptance on block sites (`foerster.block-mh/da-move`):
  `coarse` {address (fn [θ] log density)} is a cheap approximation of the
  posterior of each named block (a surrogate, a coarser model, a Laplace
  fit). Each iteration walks `:steps` random-walk steps of `:scales` on it,
  then runs the program once and accepts on the program's ratio corrected by
  the coarse one, so the chain targets the program's exact posterior; a
  poor coarse density lowers acceptance. It must be positive wherever the
  posterior is: a chain at a state of zero coarse density is refused
  (`foerster.block-mh/coarse-support`). Other latents get single-site
  random-walk moves (`:step-size`). Output options as for every chain kernel."
  [num-iterations coarse & [{:keys [scales steps step-size] :or {scales 0.1 step-size 0.1} :as opts}]]
  {:pre [(pos-int? num-iterations) (map? coarse) (seq coarse) (every? fn? (vals coarse))
         (or (nil? steps) (pos-int? steps)) (pos? step-size)
         (if (number? scales) (pos? scales) (and (seq scales) (every? pos? scales)))]}
  (chain-output (->DelayedAcceptanceKernel num-iterations coarse scales steps step-size) opts))

;; =============================================================================
;; HMCKernel
;; =============================================================================

(defrecord HMCKernel [num-iterations step-size steps]
  PMarkovKernel
  (transition-id [_] :hmc))

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

(defprotocol PSymmetricBlockKernel
  "Marks a `PBlockKernel` whose proposal is symmetric, q(x' | x) = q(x | x'):
  block Gibbs accepts its moves on the ratio of target densities alone. A
  kernel that is not symmetric has no place in block Gibbs; propose it with
  `gfi/mh-proposal` (a proposal program, scored both ways) or as an
  involutive move instead.")

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
  PSymmetricBlockKernel
  PBlockKernel
  (propose-block [_ trace block-addresses]
    (into {}
          (map (fn [addr]
                 (let [entry (get trace addr)
                       current (:value entry)]
                   ;; a Gaussian step is a move on the reals: noise on a
                   ;; discrete value is not a value of its law
                   (when-not (and (number? current) (dist/continuous? (:distribution entry)))
                     (throw (ex-info "A random-walk block kernel moves real-valued sites only"
                                     {:type ::not-real-valued :address addr})))
                   [addr (random-walk-propose current step-size)]))
               block-addresses))))

(defn random-walk-block-kernel
  [& [{:keys [step-size] :or {step-size 0.1}}]]
  {:pre [(pos? step-size)]}
  (->RandomWalkBlockKernel step-size))

(defrecord BlockGibbsKernel
           [num-iterations block-selector block-kernels address-classifier]
  PMarkovKernel
  (transition-id [_] :block-gibbs))

(defn block-gibbs-kernel
  "Create a BlockGibbsKernel for block Gibbs sampling: `address-classifier`
  (fn [address entry]) assigns latent sites to block ids, `block-selector`
  picks a block id per iteration, and `block-kernels` {block-id kernel}
  moves the block (the prior kernel, or a symmetric kernel). A move that
  changes which sites belong to the moved block is rejected: its reverse
  would move a different block.

  Output options (all Markov-chain kernels): `:samples :final` (default)
  emits each chain's last state; `:samples :all` emits every state after
  the first `:burn` moves, equally weighted — an MCMC estimate from few
  chains instead of one draw per chain."
  [num-iterations block-selector block-kernels address-classifier & [opts]]
  {:pre [(pos-int? num-iterations)
         (satisfies? PBlockSelector block-selector)
         (map? block-kernels)
         (fn? address-classifier)]}
  (when-let [bad (seq (remove (fn [[_ kernel]]
                                (or (instance? PriorBlockKernel kernel)
                                    (satisfies? PSymmetricBlockKernel kernel)))
                              block-kernels))]
    (throw (ex-info (str "Block Gibbs accepts on the target ratio alone, so a block kernel "
                         "must be the prior kernel or symmetric (PSymmetricBlockKernel)")
                    {:type ::asymmetric-block-kernel :blocks (mapv first bad)})))
  (chain-output (->BlockGibbsKernel num-iterations block-selector block-kernels address-classifier)
                opts))

(defrecord NUTSKernel [num-iterations target-accept max-depth metric step-size inverse-metric jitter]
  PMarkovKernel
  (transition-id [_] :nuts))

(defn- positive-finite? [x] (and (number? x) (pos? x) (< x ##Inf)))

(defn- per-block?
  "`x` satisfies `ok?`, or is a map from block addresses to values that do."
  [ok? x]
  (or (ok? x) (and (map? x) (seq x) (every? ok? (vals x)))))

(defn nuts-kernel
  "The No-U-Turn sampler on block sites (`foerster.nuts`), within Gibbs: the
  first `:burn` iterations adapt the step size (to `:target-accept`, default
  0.8) and a diagonal metric, and are dropped; `:max-depth` (10) bounds a
  trajectory at 2^depth leapfrog steps; `:metric` estimates the metric from
  the warm-up positions' variance (`:variance`, Stan's) or from positions
  and gradients (`:fisher`, nutpie's). A block's target must be its
  complete conditional. Output options as for every chain kernel.

  `:step-size` (default 0.1) and `:inverse-metric` (the diagonal of M⁻¹ in
  θ, default ones) set where a chain starts — each a value for every block
  site or a map {address value} — and warm-up adapts from there; with
  `:burn 0` NUTS runs frozen at them, a fixed invariant kernel, as a
  population move needs. Every retained draw's `:sampler-stats` give the
  `:step-size` and `:inverse-metric` it ran with as such maps, so a warmed-up
  run's values start a second, frozen one.

  `:jitter` (true) starts each chain driven by this kernel (`kernel-infer`,
  `infer :nuts`) at its block sites' values plus a uniform jitter in
  (−2, 2) per coordinate of θ, as Stan's random initialization; `false`
  starts exactly at `:init`. It stays on with `:burn 0`, where every draw
  is kept: resuming a frozen chain from a posterior draw (`:init` that
  draw, the exported `:step-size` and `:inverse-metric`) needs `:jitter
  false`. A population move never jitters."
  [num-iterations & [{:keys [target-accept max-depth metric step-size inverse-metric jitter]
                      :or {target-accept 0.8 max-depth 10 metric :variance jitter true} :as opts}]]
  {:pre [(pos-int? num-iterations) (#{:variance :fisher} metric)
         (boolean? jitter)
         (or (nil? step-size) (per-block? positive-finite? step-size))
         (or (nil? inverse-metric)
             (per-block? #(and (sequential? %) (seq %) (every? positive-finite? %)) inverse-metric))]}
  (chain-output (->NUTSKernel num-iterations target-accept max-depth metric step-size inverse-metric jitter)
                (select-keys opts [:samples :burn])))

;; =============================================================================
;; Programmable moves as chain components
;; =============================================================================

(defrecord InvolutiveKernel [num-iterations move]
  PMarkovKernel
  (transition-id [_] :involutive))

(defn involutive-kernel
  "A Markov-chain kernel making `num-iterations` involutive MH moves
  (`foerster.involutive/step`): `move` is {:propose :log-q :involution},
  optionally :check?. It composes with the other chain kernels through
  `cycle` and `mixture`. Chains only: population programs refuse it, since
  it gives back the state it leaves. Output options as for every chain
  kernel."
  [num-iterations move & [opts]]
  {:pre [(pos-int? num-iterations) (fn? (:propose move)) (fn? (:log-q move)) (fn? (:involution move))]}
  (chain-output (->InvolutiveKernel num-iterations move) opts))

(defrecord ProposalMHKernel [num-iterations proposal args]
  PMarkovKernel
  (transition-id [_] :proposal-mh))

(defn proposal-mh-kernel
  "A Markov-chain kernel making `num-iterations` MH moves with the proposal
  program `proposal` (`foerster.gfi/mh-proposal`: a function of the current
  choices and `:args` returning a spin that proposes model addresses). It
  composes through `cycle` and `mixture`; chains only, as
  `involutive-kernel`. Output options as for every chain kernel."
  [num-iterations proposal & [{:keys [args] :as opts}]]
  {:pre [(pos-int? num-iterations) (fn? proposal)]}
  (chain-output (->ProposalMHKernel num-iterations proposal (vec args)) (dissoc opts :args)))

;; =============================================================================
;; Composing Markov-chain kernels
;; =============================================================================

(defn- check-chain-kernels! [kernels]
  (when-let [bad (seq (remove markov-kernel? kernels))]
    (throw (ex-info "Only Markov-chain kernels compose"
                    {:type ::not-a-chain-kernel :kernels (vec bad)}))))

(defrecord CycleKernel [num-iterations kernels]
  PMarkovKernel
  (transition-id [_] :cycle))

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
  PMarkovKernel
  (transition-id [_] :mixture))

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
