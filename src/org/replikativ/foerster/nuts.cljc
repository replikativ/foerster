(ns org.replikativ.foerster.nuts
  "The No-U-Turn sampler on block sites (`foerster.block`), within Gibbs, as
  Stan implements it: multinomial sampling over the trajectory, the
  generalized no-U-turn criterion with the checks across merged subtrees
  (Hoffman & Gelman 2014; Betancourt 2017), divergences when the energy error
  exceeds 1000, a diagonal metric, and during warm-up dual-averaging step
  size adaptation and windowed variance estimation of the metric (Stan's
  75/25/50 windows, scaled to short warm-ups). Stan's step size heuristic
  (`init_stepsize`: double or halve until one leapfrog step's acceptance
  crosses 0.8) runs as warm-up begins and after every metric window, and
  dual averaging restarts from it, at μ = log 10ε. As warm-up begins this
  deliberately deviates from Stan, whose first μ is log 10ε of its
  configured step size (default 1): here it comes from the heuristic's ε,
  as at every later restart, so the first window does not start from a
  scale-blind guess.

  NUTS samples the block's own density, so the block's target must be the
  complete conditional of its latents (`:block/target
  :complete-conditional`): every factor they touch inside the block, and no
  other site depending on them. A replay that changes the log probability
  of anything outside the block is refused (`::incomplete-target`, its
  `:changed` the sites whose score moved). Put a hierarchy into one block,
  or move such a block with HMC (`foerster.hmc`) or delayed acceptance
  (`foerster.block-mh`), which accept on the full trace (doc/algorithms.md,
  section NUTS within Gibbs needs complete conditionals).

  Each chain starts at its block sites' `:init` plus U(−2, 2) per
  coordinate of θ (`jitter`), unless `:jitter false` — also with `:burn
  0`, where every draw is kept: to resume a frozen chain from a posterior
  draw, pass that draw as `:init` with `:jitter false`. `:step-size` and
  `:inverse-metric` set where warm-up starts, or with `:burn 0` the frozen
  values NUTS runs at; the sampler statistics report them.

  The kernel is `(k/nuts-kernel iterations {:burn … :target-accept 0.8
  :max-depth 10})`: the first `:burn` iterations adapt and are dropped."
  (:require [org.replikativ.foerster.transition-context :as tc]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.trace :as trace]))

;; =============================================================================
;; Vectors — double arrays inside a transition. Every operation allocates its
;; result and sums left to right from 0.0, as the sequence versions did, so
;; the draws are the same numbers.
;; =============================================================================

(defn- dot ^double [^doubles a ^doubles b]
  (let [n (alength a)]
    (loop [i 0 acc 0.0] (if (= i n) acc (recur (inc i) (+ acc (* (aget a i) (aget b i))))))))

(defn- v+ ^doubles [^doubles a ^doubles b]
  (let [n (alength a) out (double-array n)]
    (dotimes [i n] (aset out i (+ (aget a i) (aget b i))))
    out))

(defn- axpy "a·x + y" ^doubles [a ^doubles x ^doubles y]
  (let [a (double a) n (alength x) out (double-array n)]
    (dotimes [i n] (aset out i (+ (* a (aget x i)) (aget y i))))
    out))

(defn- emul "elementwise a·b" ^doubles [^doubles a ^doubles b]
  (let [n (alength a) out (double-array n)]
    (dotimes [i n] (aset out i (* (aget a i) (aget b i))))
    out))

(defn- finite? [x] (not (or (NaN? x) (infinite? x))))

(defn- all-finite? [^doubles a]
  (let [n (alength a)]
    (loop [i 0] (cond (= i n) true (finite? (aget a i)) (recur (inc i)) :else false))))

(defn- log-sum-exp [a b]
  (cond (= ##-Inf a) b
        (= ##-Inf b) a
        :else (let [top (max a b)] (+ top (Math/log (+ (Math/exp (- a top)) (Math/exp (- b top))))))))

;; =============================================================================
;; The trajectory
;; =============================================================================

(defn- point
  "A phase-space point: position, momentum, log density and gradient, the
  last two given as `start` ([lp g]) or evaluated."
  [dist inv-metric q p start]
  (let [[lp g] (or start (block/value+grad! dist q))]
    {:q q :p p :lp lp :g g :p-sharp (emul inv-metric p)}))

(defn- hamiltonian [{:keys [lp p p-sharp]}]
  (- (* 0.5 (dot p p-sharp)) lp))

(defn- leapfrog [dist inv-metric eps direction {:keys [q p g]}]
  (let [e (* direction eps)
        p (axpy (* 0.5 e) g p)
        q (axpy e (emul inv-metric p) q)
        [lp g'] (if (all-finite? q) (block/value+grad! dist q) [##-Inf (double-array (alength ^doubles q))])
        p (axpy (* 0.5 e) g' p)]
    {:q q :p p :lp lp :g g' :p-sharp (emul inv-metric p)}))

(defn- criterion
  "The generalized no-U-turn criterion: the trajectory still extends in both
  directions along the summed momentum ρ."
  [p-sharp-minus p-sharp-plus rho]
  (and (> (dot p-sharp-plus rho) 0.0) (> (dot p-sharp-minus rho) 0.0)))

(defn- build-tree
  "A subtree of 2^depth leapfrog steps from `z` in `direction` (Stan's
  `build_tree`). Returns {:valid? :z-end :z-propose :log-w :rho :beg :end
  :n :metro :divergent?}; `:beg`/`:end` are its first and last points in the
  direction of travel."
  [dist inv-metric eps direction depth z h0]
  (if (zero? depth)
    (let [z' (leapfrog dist inv-metric eps direction z)
          h (let [h (hamiltonian z')] (if (NaN? h) ##Inf h))
          divergent? (> (- h h0) 1000.0)]
      (if divergent?
        {:valid? false :divergent? true :n 1 :metro 0.0}
        {:valid? true :z-end z' :z-propose z' :log-w (- h0 h) :rho (:p z')
         :beg z' :end z' :n 1 :metro (min 1.0 (Math/exp (- h0 h)))}))
    (let [left (build-tree dist inv-metric eps direction (dec depth) z h0)]
      (if-not (:valid? left)
        left
        (let [right (build-tree dist inv-metric eps direction (dec depth) (:z-end left) h0)
              n (+ (:n left) (:n right))
              metro (+ (:metro left) (:metro right))]
          (if-not (:valid? right)
            (assoc right :n n :metro metro)
            (let [log-w (log-sum-exp (:log-w left) (:log-w right))
                  z-propose (if (or (> (:log-w right) log-w)
                                    (< (random/uniform01) (Math/exp (- (:log-w right) log-w))))
                              (:z-propose right)
                              (:z-propose left))
                  rho (v+ (:rho left) (:rho right))
                  persist? (and (criterion (:p-sharp (:beg left)) (:p-sharp (:end right)) rho)
                                ;; across the merge (Stan ≥ 2.23)
                                (criterion (:p-sharp (:beg left)) (:p-sharp (:beg right))
                                           (v+ (:rho left) (:p (:beg right))))
                                (criterion (:p-sharp (:end left)) (:p-sharp (:end right))
                                           (v+ (:rho right) (:p (:end left)))))]
              {:valid? persist? :z-end (:z-end right) :z-propose z-propose :log-w log-w :rho rho
               :beg (:beg left) :end (:end right) :n n :metro metro})))))))

(defn- momentum
  "A momentum p ~ N(0, M), M the inverse of `inv-metric`."
  ^doubles [^doubles inv-metric]
  (let [n (alength inv-metric) out (double-array n)]
    (dotimes [i n]
      (aset out i (* (double (dist/draw (dist/normal 0.0 1.0))) (Math/sqrt (/ 1.0 (aget inv-metric i))))))
    out))

(defn transition
  "One NUTS transition of `dist` (a block distribution) from position `q0`
  with step size `eps` and diagonal inverse metric `inv-metric`, drawing from
  the current generator. `start`, when given, is [log density, gradient] at
  q0 (the previous transition's), which is then not evaluated again.
  Returns {:q :lp (log density at q) :g (its gradient) :accept-stat :depth
  :n-leapfrog :divergent?}."
  ([dist q0 eps inv-metric max-depth] (transition dist q0 eps inv-metric max-depth nil))
  ([dist q0 eps inv-metric max-depth start]
   (let [inv-metric (double-array inv-metric)
         q0 (double-array q0)
         p0 (momentum inv-metric)
         z0 (point dist inv-metric q0 p0 (when start [(double (first start)) (double-array (second start))]))
         h0 (hamiltonian z0)
         result (fn [sample n metro depth divergent?]
                  {:q (vec (:q sample)) :lp (:lp sample) :g (vec (:g sample)) :accept-stat (/ metro (max 1 n))
                   :depth depth :n-leapfrog n :divergent? divergent?})]
     (loop [depth 0 fwd z0 bck z0 sample z0 log-w 0.0 rho (:p z0) n 0 metro 0.0 divergent? false]
       (if (>= depth max-depth)
         (result sample n metro depth divergent?)
         (let [forward? (> (random/uniform01) 0.5)
               from (if forward? fwd bck)
               sub (build-tree dist inv-metric eps (if forward? 1 -1) depth from h0)
               n (+ n (:n sub)) metro (+ metro (:metro sub))]
           (if-not (:valid? sub)
             ;; Stan counts a subtree in the depth only when it is valid
             (result sample n metro depth (boolean (:divergent? sub)))
             (let [sample (if (or (> (:log-w sub) log-w)
                                  (< (random/uniform01) (Math/exp (- (:log-w sub) log-w))))
                            (:z-propose sub)
                            sample)
                   log-w (log-sum-exp log-w (:log-w sub))
                   [fwd bck] (if forward? [(:end sub) bck] [fwd (:end sub)])
                  ;; the whole trajectory's ends: backward end, forward end
                   [bck-end fwd-end] [bck fwd]
                   rho' (v+ rho (:rho sub))
                  ;; sub's first point sits next to the old end it grew from
                   [old-rho new-rho] [rho (:rho sub)]
                   persist? (and (criterion (:p-sharp bck-end) (:p-sharp fwd-end) rho')
                                 (if forward?
                                   (and (criterion (:p-sharp bck-end) (:p-sharp (:beg sub)) (v+ old-rho (:p (:beg sub))))
                                        (criterion (:p-sharp from) (:p-sharp fwd-end) (v+ new-rho (:p from))))
                                   (and (criterion (:p-sharp (:beg sub)) (:p-sharp fwd-end) (v+ old-rho (:p (:beg sub))))
                                        (criterion (:p-sharp bck-end) (:p-sharp from) (v+ new-rho (:p from))))))]
               (if persist?
                 (recur (inc depth) fwd bck sample log-w rho' n metro divergent?)
                 (result sample n metro (inc depth) divergent?))))))))))

;; =============================================================================
;; Adaptation (Stan's defaults)
;; =============================================================================

(def ^:private log-0-8 (Math/log 0.8))

(defn- init-step-size
  "Stan's `init_stepsize` (base_hmc.hpp): from `eps`, double the step size
  while one leapfrog step from `q` with fresh momentum is accepted with
  probability above 0.8, or halve it while it is below, whichever the first
  step says, until that crosses. `start` is [log density, gradient] at `q`.
  An extreme `eps` (0 or above 1e7) is returned as it is."
  [dist inv-metric q start eps]
  (let [inv-metric (double-array inv-metric)
        q (double-array q)
        start [(double (first start)) (double-array (second start))]
        delta-h (fn [eps]
                  (let [z (point dist inv-metric q (momentum inv-metric) start)
                        h0 (hamiltonian z)
                        h (let [h (hamiltonian (leapfrog dist inv-metric eps 1 z))] (if (NaN? h) ##Inf h))]
                    (- h0 h)))]
    (if (or (zero? eps) (> eps 1e7))
      eps
      (let [direction (if (> (delta-h eps) log-0-8) 1 -1)]
        (loop [eps eps]
          (let [dh (delta-h eps)]
            (if (if (= 1 direction) (not (> dh log-0-8)) (not (< dh log-0-8)))
              eps
              (let [eps (if (= 1 direction) (* 2.0 eps) (* 0.5 eps))]
                (when (> eps 1e7)
                  (throw (ex-info "Posterior is improper: no step size is too large for it"
                                  {:type ::improper-posterior :step-size eps})))
                (when (zero? eps)
                  (throw (ex-info "No acceptably small step size: is the log density continuous?"
                                  {:type ::no-step-size})))
                (recur eps)))))))))

(defn- da-init [eps] {:mu (Math/log (* 10.0 eps)) :s-bar 0.0 :x-bar 0.0 :counter 0})

(defn- da-update
  "Dual averaging (Nesterov 2009; Hoffman & Gelman 2014): the next log step
  size from the accept statistic. Returns [state eps]."
  [{:keys [mu s-bar x-bar counter]} accept-stat target]
  (let [counter (inc counter)
        eta (/ 1.0 (+ counter 10.0))
        s-bar (+ (* (- 1.0 eta) s-bar) (* eta (- target accept-stat)))
        x (- mu (/ (* s-bar (Math/sqrt counter)) 0.05))
        x-eta (Math/pow counter -0.75)
        x-bar (+ (* (- 1.0 x-eta) x-bar) (* x-eta x))]
    [{:mu mu :s-bar s-bar :x-bar x-bar :counter counter} (Math/exp x)]))

(defn- windows
  "The metric adaptation windows [start end) of a warm-up of `burn`
  iterations: Stan's initial 75 and terminal 50 buffers around windows
  doubling from 25, scaled down for short warm-ups (none below 20)."
  [burn]
  (if (< burn 20)
    []
    (let [[init term base] (if (< burn 150)
                             (let [i (long (* 0.15 burn)) t (long (* 0.1 burn))] [i t (- burn i t)])
                             [75 50 25])
          end (- burn term)]
      (loop [start init size base out []]
        (if (>= start end)
          out
          (let [stop (+ start size)
                ;; the last window stretches to the terminal buffer
                stop (if (> (+ stop (* 2 size)) end) end stop)]
            (recur stop (* 2 size) (conj out [start stop]))))))))

(defn- regularized-variance [xs]
  (let [n (count xs) d (count (first xs))
        mean (mapv #(/ (reduce + (map (fn [x] (nth x %)) xs)) n) (range d))]
    (mapv (fn [j] (let [v (/ (reduce + (map #(let [e (- (nth % j) (nth mean j))] (* e e)) xs)) (dec n))]
                    (+ (* (/ n (+ n 5.0)) v) (* 1e-3 (/ 5.0 (+ n 5.0))))))
          (range d))))

(defn- variance [xs j]
  (let [n (count xs) mean (/ (reduce + (map #(nth % j) xs)) n)]
    (/ (reduce + (map #(let [e (- (nth % j) mean)] (* e e)) xs)) (dec n))))

(defn- fisher-variance
  "The diagonal inverse metric minimizing the Fisher divergence between the
  target and its Gaussian approximation, from positions `qs` and the log
  density's gradients `gs` there: √(Var q_j / Var g_j) per coordinate
  (Seyboldt, Carlson & Carpenter 2024, nutpie). Exact with a Gaussian
  target's draws; a coordinate whose gradient does not vary keeps its
  position variance."
  [qs gs]
  (mapv (fn [j]
          (let [vq (variance qs j) vg (variance gs j)]
            (min 1e10 (max 1e-10 (if (pos? vg) (Math/sqrt (/ vq vg)) vq)))))
        (range (count (first qs)))))

(defn- adapt
  "The adaptation state after warm-up iteration `i`, which ended at position
  `q` with gradient `g` and accept statistic `stat`: the step size by dual
  averaging; within a window the positions (and gradients) are collected,
  and at its end they give the metric — their regularized variance, or the
  Fisher estimate with `metric` :fisher — and the step size restarts
  (`:restart?`: `nuts-move` runs Stan's heuristic from the new metric)."
  [{:keys [da windows draws metric] :as st} i q g stat target]
  (let [[da eps] (da-update da stat target)
        st (assoc st :da da :eps eps)
        window (first (filter (fn [[a b]] (and (<= a i) (< i b))) windows))
        draws' (conj draws [q g])]
    (cond
      (nil? window) st
      (= (inc i) (second window))
      (assoc st
             :inv-metric (if (= metric :fisher)
                           (fisher-variance (mapv first draws') (mapv second draws'))
                           (regularized-variance (mapv first draws')))
             :draws [] :restart? true)
      :else (assoc st :draws draws'))))

;; =============================================================================
;; Within Gibbs
;; =============================================================================

(defn- block-site? [trace address]
  (block/block-dist? (get-in trace [:trace/entries address :note :dist])))

(defn- initial-state
  "A chain's adaptation state for the block site at `address` of dimension
  `dim`, starting from the given `step-size` and `inverse-metric` (each a
  value or a map {address value}), else ε = 0.1 and the unit metric."
  [address dim burn metric step-size inverse-metric]
  (let [for-block #(if (map? %) (get % address) %)
        eps (double (or (for-block step-size) 0.1))
        inv-metric (if-let [v (for-block inverse-metric)]
                     (if (= dim (count v))
                       (mapv double v)
                       (throw (ex-info "The inverse metric's length is not the block's dimension"
                                       {:type ::inverse-metric-dimension :address address
                                        :dimension dim :length (count v)})))
                     (vec (repeat dim 1.0)))]
    {:eps eps :da (da-init eps) :inv-metric inv-metric
     :windows (windows burn) :draws [] :metric metric}))

(defn- nuts-move
  "Move the block site at `address` by one NUTS transition and replay; the
  chain's adaptation state is in the atom `adaptation`."
  [trace address iteration {:keys [burn target-accept max-depth metric adaptation-iteration
                                   step-size inverse-metric]
                            :as context} adaptation]
  (fn [resolve reject]
    (try
      (tc/require-untempered! context :nuts)
      (let [local-iteration (or adaptation-iteration iteration)
            {q0 :value {dist :dist} :note} (get-in trace [:trace/entries address])
            dim (count q0)
            world (:trace/world trace)
            warm-up? (< local-iteration burn)
            fresh? (nil? (get @adaptation address))
            st (or (get @adaptation address)
                   (initial-state address dim burn metric step-size inverse-metric))
            ;; the previous transition's end, when the chain is still there
            ;; and the block's inputs have not moved since
            start (let [{:keys [q lp g] :as end} (:end st)]
                    (when (and end (identical? (:block dist) (:block end))
                               (= (:inputs dist) (:inputs end)) (= q q0))
                      [lp g]))
            ;; Stan's step size heuristic as warm-up begins; dual averaging
            ;; then shrinks towards 10 times the step size it found
            [st start] (if (and fresh? warm-up?)
                         (let [start (or start (block/value+grad! dist (double-array q0)))
                               eps (random/in-world-stream world [::step-size address iteration]
                                                           #(init-step-size dist (:inv-metric st) q0 start (:eps st)))]
                           [(assoc st :eps eps :da (da-init eps)) start])
                         [st start])
            {:keys [q lp g accept-stat divergent?] :as tr}
            (random/in-world-stream world [::transition address iteration]
                                    #(transition dist q0 (:eps st) (:inv-metric st) max-depth start))
            st (assoc st :end {:block (:block dist) :inputs (:inputs dist) :q q :lp lp :g g})
            st' (if warm-up?
                  (let [s (adapt st local-iteration q g accept-stat target-accept)
                        ;; a metric window ended: Stan's heuristic again from
                        ;; the last step size under the new metric, and dual
                        ;; averaging restarts from it
                        s (if (:restart? s)
                            (let [eps (random/in-world-stream world [::step-size address iteration]
                                                              #(init-step-size dist (:inv-metric s) q [lp g] (:eps s)))]
                              (-> s (dissoc :restart?) (assoc :eps eps :da (da-init eps))))
                            s)]
                    ;; warm-up over: the averaged step size
                    (cond-> s
                      (= (inc local-iteration) burn) (assoc :eps (Math/exp (:x-bar (:da s))))))
                  st)]
        (swap! adaptation assoc address st')
        (let [tr (assoc tr :step-size (:eps st) :inverse-metric (:inv-metric st))]
          (if (= q q0)
            (resolve {:trace trace :accepted? false :divergent? divergent? :stat tr})
            (let [move (tc/policy context (fn [sp _]
                                            ;; the replay scores q by the block's
                                            ;; :log-density, never by :value+grad's lp:
                                            ;; that need only be right up to a constant
                                            ;; in θ, which may depend on the inputs
                                            (when (= address (:savepoint/address sp))
                                              {:value q :symmetric? true})))]
              ((itrace/reexecute trace #{address} move (tc/replay-options context trace [::move address iteration]))
               (fn [proposed]
                 (try
                   (when (or (:trace/error proposed) (not (tc/at-frontier? context proposed)))
                     (trace/release! proposed trace)
                     (throw (ex-info "NUTS replay failed or left its declared frontier"
                                     {:type ::invalid-replay :address address} (:trace/error proposed))))
                   (let [entry-score (tc/entry-score context)
                         joint (- (tc/score context proposed) (tc/score context trace))
                         own (- (entry-score (get-in proposed [:trace/entries address]))
                                (entry-score (get-in trace [:trace/entries address])))]
                     (when (or (:trace/error proposed) (> (Math/abs (double (- joint own))) 1e-6))
                       (let [e (tc/incomplete-target "NUTS" ::incomplete-target context trace proposed address (- joint own))]
                         (trace/release! proposed trace)
                         (throw e)))
                     (tc/settle! context trace proposed true)
                     (resolve {:trace proposed :accepted? true :divergent? divergent? :stat tr}))
                   (catch #?(:clj Throwable :cljs :default) e (reject e))))
               reject)))))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

(def ^:private jitter-attempts 100)

(defn- jittered
  "θ + U(−2, 2) per coordinate whose log density and gradient are finite —
  Stan's initialization radius and its 100 attempts, keyed to `world`'s
  stream — or nil when no attempt is."
  [world address dist theta]
  (let [u (dist/uniform -2.0 2.0)]
    (some (fn [attempt]
            (let [theta' (random/in-world-stream world [::jitter address attempt]
                                                 #(mapv (fn [x] (+ (double x) (double (dist/draw u)))) theta))
                  [lp g] (block/value+grad! dist (double-array theta'))]
              (when (and (finite? lp) (all-finite? g)) theta')))
          (range jitter-attempts))))

(defn jitter
  "A chain's start for NUTS: every block site of `trace` (the chain's
  initial state) moved by a uniform jitter in (−2, 2) per coordinate of θ,
  the unconstrained coordinates — Stan's random initialization radius,
  around the site's `:init` as PyMC jitters around its initial point — so
  that chains start apart and R̂ can see them disagree. The draws come from
  the chain's world stream, so a seeded run is reproducible. A site keeps
  its value when none of 100 jitters has a finite log density and
  gradient, and the whole state keeps it when the jittered state has zero
  density. Returns a CPS operation resolving the new trace."
  [trace]
  (fn [resolve reject]
    (try
      (let [world (:trace/world trace)
            values (into {}
                         (keep (fn [address]
                                 (let [{theta :value {dist :dist} :note} (get-in trace [:trace/entries address])]
                                   (some->> (jittered world address dist theta) (vector address)))))
                         (filterv #(block-site? trace %) (itrace/latent-addresses trace)))]
        (if (empty? values)
          (resolve trace)
          (let [move (tc/policy {} (fn [sp _]
                                     (when-let [v (get values (:savepoint/address sp))]
                                       {:value v :symmetric? true})))]
            ((itrace/reexecute trace (set (keys values)) move (tc/replay-options {} trace [::jitter]))
             (fn [proposed]
               (try
                 (if (or (:trace/error proposed) (not (finite? (itrace/log-joint proposed))))
                   (do (trace/release! proposed trace) (resolve trace))
                   (do (tc/settle! {} trace proposed true) (resolve proposed)))
                 (catch #?(:clj Throwable :cljs :default) e (reject e))))
             reject))))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

(defn within-gibbs
  "A step for `foerster.trace/mh-chain` (`:step`): a NUTS move of every block
  site, then one single-site MH move of a latent that is not a block site, if
  there is one. A fresh adaptation state per call (per chain). `opts`:
  `:burn`, `:target-accept` (0.8), `:max-depth` (10), `:metric` (`:variance`,
  Stan's estimate from the positions, or `:fisher`, from positions and
  gradients), `:step-size` (0.1) and `:inverse-metric` (ones), where each
  block's adaptation starts, or, with `:burn` 0, the fixed values it runs
  with — each one value for every block site or a map {address value}.
  The sampler statistics' `:step-size` and `:inverse-metric` are such maps,
  the values each block's transition ran with.
  Returns per-step `:sampler-stats` for direct NUTS inference. The flag
  `:reached-max-depth?` reports depth attainment, including transitions that
  stop naturally or diverge at that depth; it does not identify cap exhaustion.
  Composed kernels and final-state-only output do not retain these statistics."
  [{:keys [burn target-accept max-depth metric step-size inverse-metric]
    :or {burn 0 target-accept 0.8 max-depth 10 metric :variance}}]
  (let [adaptation (atom {})
        divergences (atom 0)]
    (fn [trace {:keys [iteration] :as chain-opts}]
      (fn [resolve reject]
        (let [opts (assoc chain-opts :burn burn :target-accept target-accept :max-depth max-depth :metric metric
                          :step-size step-size :inverse-metric inverse-metric)
              blocks (filterv #(block-site? trace %) (itrace/latent-addresses trace))]
          (letfn [(others [t] (vec (remove #(block-site? t %) (itrace/latent-addresses t))))
                  (go [t [address & more] moves stats]
                      (if address
                        ((nuts-move t address iteration opts adaptation)
                         (fn [r]
                           (when-let [visit (:on-visit chain-opts)] (visit r))
                           (when (and (:divergent? r) (>= (or (:adaptation-iteration opts) iteration) burn)) (swap! divergences inc))
                           (go (:trace r) more (inc moves)
                               (conj stats (assoc (select-keys (:stat r) [:accept-stat :depth :n-leapfrog
                                                                          :step-size :inverse-metric])
                                                  :address address
                                                  :divergent? (boolean (:divergent? r))))))
                         reject)
                        (let [done (fn [t moves accepted]
                                     (resolve {:trace t :accepted? (pos? accepted) :moves moves
                                               :accepted-moves accepted :divergences @divergences
                                               :sampler-stats
                                               {:diverging (boolean (some :divergent? stats))
                                                :tree-depth (reduce max 0 (map :depth stats))
                                                :n-leapfrog (reduce + 0 (map :n-leapfrog stats))
                                                :accept-stat (/ (reduce + 0.0 (map :accept-stat stats))
                                                                (max 1 (count stats)))
                                                :reached-max-depth? (boolean (some #(>= (:depth %) max-depth) stats))
                                                :step-size (into {} (map (juxt :address :step-size)) stats)
                                                :inverse-metric (into {} (map (juxt :address :inverse-metric)) stats)
                                                :blocks (count stats)}}))]
                          (if (seq (others t))
                            ((itrace/mh-step t (assoc chain-opts
                                                      :select (fn [t' _]
                                                                {:targets #{(m/pick-uniformly (others t'))}
                                                                 :log-selection #(- (Math/log (double (count (others %)))))})))
                             (fn [r]
                               (when-let [visit (:on-visit chain-opts)] (visit r))
                               (done (:trace r) (inc moves) (+ moves (if (:accepted? r) 1 0))))
                             reject)
                            (done t moves moves)))))]
            (go trace blocks 0 [])))))))
