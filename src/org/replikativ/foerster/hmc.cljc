(ns org.replikativ.foerster.hmc
  "Hamiltonian Monte Carlo on block sites (`foerster.block`), within Gibbs.

  A block supplies the gradient of its log target; the move is spindel's. The
  momentum is drawn from the site's stream, leapfrog integrates with the
  block's `:value+grad`, and the endpoint is proposed by replaying the
  computation from the block site. It is accepted on the change of the FULL
  trace log joint plus the kinetic energy, not on the block's own value:

    log α = [log p(trace') − K(p')] − [log p(trace) − K(p)] + log q(rev)/q(fwd)

  where the last term, as in single-site MH (`trace/mh-log-ratio`), covers
  the sites the replay drew afresh or no longer reaches.

  Leapfrog is volume preserving and reversible for any position-dependent
  force, so the move is exact whatever the block's target covers. A block
  whose target misses factors its latents affect (a downstream observe, a
  dependent site) mixes worse, and the step says so: `:incomplete-target?`
  is true when the replay changed the log probability of anything outside the
  block."
  (:require [org.replikativ.spindel.trace :as trace]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.dist :as dist]))

(defn- kinetic [p] (* 0.5 (reduce + 0.0 (map #(* % %) p))))

(defn- axpy "a·x + y" [a x y] (mapv #(+ (* a %1) %2) x y))

(defn- finite-vector? [v] (every? #(not (or (NaN? %) (infinite? %))) v))

(defn- leapfrog
  "`steps` leapfrog steps of size `eps` from (q, p) under the block's force."
  [dist q p eps steps]
  (let [grad #(second (block/value+grad dist %))]
    (loop [q q
           p (axpy (* 0.5 eps) (grad q) p)
           i 1]
      (let [q (axpy eps p q)]
        (cond
          (not (finite-vector? q)) [q p]
          (= i steps) [q (axpy (* 0.5 eps) (grad q) p)]
          :else (recur q (axpy eps (grad q) p) (inc i)))))))

(defn block-site?
  "Whether the trace entry at `address` is a block site."
  [trace address]
  (block/block-dist? (get-in trace [:trace/entries address :note :dist])))

(defn- entry-log-prob [trace address]
  (get-in trace [:trace/entries address :note :log-prob]))

(defn hmc-step
  "One HMC move of the block site at `address` of `trace`.

  Options: `:step-size`, `:steps` (leapfrog steps), `:iteration`,
  `:constraints` (the chain's conditioning, as for `mh-step`). Returns a CPS
  operation resolving {:trace :accepted? :log-ratio :incomplete-target?}."
  [trace {:keys [address step-size steps iteration constraints]
          :or {step-size 0.1 steps 10 iteration 0}}]
  (fn [resolve reject]
    (try
      (let [{q0 :value {dist :dist} :note} (get-in trace [:trace/entries address])
            _ (when-not (block/block-dist? dist)
                (throw (ex-info "HMC moves block sites only"
                                {:type ::not-a-block :address address})))
            p0 (random/in-world-stream
                (:trace/world trace) [::momentum address iteration]
                #(vec (repeatedly (count q0) (fn [] (dist/draw (dist/normal 0.0 1.0))))))
            [q1 p1] (leapfrog dist q0 p0 step-size steps)]
        (if-not (finite-vector? q1)
          (resolve {:trace trace :accepted? false :log-ratio ##-Inf :incomplete-target? false})
          (let [move (itrace/policy
                      {:keep? true
                       :constraints constraints
                       :draw (fn [sp _]
                               (when (= address (:savepoint/address sp))
                                 {:value q1 :symmetric? true}))})]
            ((trace/replay trace address move {:anchor? itrace/anchor?})
             (fn [proposed]
               (try
                 (let [joint-delta (if (:trace/error proposed)
                                     ##-Inf
                                     (- (itrace/log-joint proposed) (itrace/log-joint trace)))
                       block-delta (- (or (entry-log-prob proposed address) ##-Inf)
                                      (entry-log-prob trace address))
                       ;; the full MH ratio of the replay — sites it drew
                       ;; afresh or dropped enter as in single-site MH — plus
                       ;; the kinetic energy of the leapfrog endpoint
                       ratio (if (:trace/error proposed)
                               ##-Inf
                               (+ (itrace/mh-log-ratio trace proposed (constantly 0.0) address)
                                  (- (kinetic p0) (kinetic p1))))
                       accept? (and (not (NaN? ratio))
                                    (or (>= ratio 0.0)
                                        (< (Math/log (random/in-world-stream
                                                      (:trace/world proposed) ::accept
                                                      m/uniform01))
                                           ratio)))]
                   (if accept?
                     (trace/release! trace proposed)
                     (trace/release! proposed trace))
                   (resolve {:trace (if accept? proposed trace)
                             :accepted? accept?
                             :log-ratio ratio
                             :incomplete-target? (and (not (infinite? joint-delta))
                                                      (> (Math/abs (- joint-delta block-delta)) 1e-9))}))
                 (catch #?(:clj Throwable :cljs :default) error
                   (reject error))))
             reject))))
      (catch #?(:clj Throwable :cljs :default) error
        (reject error)))))

(defn within-gibbs
  "A step for `foerster.trace/mh-chain` (`:step`): an HMC move of every
  block site, then one single-site MH move of a latent that is not a block
  site, if there is one. `opts`: `:step-size`, `:steps`. Resolves the step
  with `:moves` (how many moves it made) and `:accepted-moves`; the step as a
  whole is `:accepted?` when any of them was."
  [{:keys [step-size steps] :as opts}]
  (fn [trace {:keys [iteration] :as chain-opts}]
    (fn [resolve reject]
      (let [blocks (filterv #(block-site? trace %) (itrace/latent-addresses trace))]
        (letfn [(others [t] (vec (remove #(block-site? t %) (itrace/latent-addresses t))))
                (go [t [address & more] moves accepted incomplete?]
                    (if address
                      ((hmc-step t (merge chain-opts opts {:address address :iteration iteration
                                                           :step-size step-size :steps steps}))
                       (fn [r] (go (:trace r) more (inc moves) (if (:accepted? r) (inc accepted) accepted)
                                   (or incomplete? (:incomplete-target? r))))
                       reject)
                      (let [done (fn [t moves accepted]
                                   (resolve {:trace t :accepted? (pos? accepted)
                                             :moves moves :accepted-moves accepted
                                             :incomplete-target? incomplete?}))]
                        (if (seq (others t))
                          ((itrace/mh-step t (assoc chain-opts
                                                    :select (fn [t' _]
                                                              {:targets #{(m/pick-uniformly (others t'))}
                                                               :log-selection #(- (Math/log (double (count (others %)))))})))
                           (fn [r] (done (:trace r) (inc moves) (if (:accepted? r) (inc accepted) accepted)))
                           reject)
                          (done t moves accepted)))))]
          (go trace blocks 0 0 false))))))
