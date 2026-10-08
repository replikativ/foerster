(ns org.replikativ.foerster.block-mh
  "Random-walk Metropolis on block sites (`foerster.block`), run on the
  block's θ vector: a move takes several Gaussian steps against the block's
  own density and writes the result back to the trace once, as NUTS does
  (`foerster.nuts`), so the trace replay is paid per move rather than per
  density evaluation.

  The density is the block's target, or at temperature β the geometric path
  q^(1-β)·t^β from its draw density q to its target t (tempered SMC,
  `foerster.tempering`). The block's target must be the complete conditional
  of its latents: a write-back that changes the log probability of anything
  outside the block is refused (`::incomplete-target`).

  `within-gibbs` is the step of the random-walk MH kernel: block sites move
  this way, with per-coordinate scales adapted during the chain's `:burn`
  iterations (a diagonal adaptive Metropolis whose overall scale steers the
  acceptance rate toward 0.234), then one single-site random-walk move of a
  latent that is not a block site, if there is one. A trace without block
  sites makes the same moves as before."
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.trace :as trace]))

(defn block-site? [trace address]
  (block/block-dist? (get-in trace [:trace/entries address :note :dist])))

(defn density
  "The log density a move of a block site with law `dist` targets: its target
  t, or at temperature `beta` the path q^(1-β)·t^β from its draw density q."
  [dist beta]
  (if (nil? beta)
    #(dist/logpdf dist %)
    (fn [theta]
      (let [lq (dist/-draw-logpdf dist theta)]
        (if (zero? beta) lq (+ lq (* beta (- (dist/logpdf dist theta) lq))))))))

(defn walk
  "`steps` random-walk Metropolis steps from `q0` against the log density `f`,
  coordinate j stepping by `scales`[j] standard normals. Resolves
  {:q final :accepted k}."
  [f q0 scales steps]
  (loop [i 0 q (vec q0) lp (f q0) accepted 0]
    (if (= i steps)
      {:q q :accepted accepted}
      (let [q' (mapv (fn [x s] (+ x (* s (dist/draw (dist/normal 0.0 1.0))))) q scales)
            lp' (f q')
            u (dist/draw (dist/uniform 0.0 1.0))]
        (if (and (== lp' lp') (< (Math/log u) (- lp' lp)))
          (recur (inc i) q' lp' (inc accepted))
          (recur (inc i) q lp accepted))))))

(defn- write-back
  "Replay `trace` with the block site at `address` set to `q` (a symmetric
  move, at `beta` when tempering); resolves the new trace. The old trace's
  worlds are given back unless `keep-old?` (its owner releases them, as
  tempered SMC does for particles that share an ancestor)."
  [trace address q beta keep-old?]
  (fn [resolve reject]
    (let [move (itrace/policy (cond-> {:keep? true
                                       :draw (fn [sp _] (when (= address (:savepoint/address sp))
                                                          {:value q :symmetric? true}))}
                                beta (assoc :temperature beta)))]
      ((trace/replay trace address move {:anchor? itrace/anchor?})
       (fn [proposed]
         (try
           (let [joint (- (itrace/log-joint proposed) (itrace/log-joint trace))
                 own (- (get-in proposed [:trace/entries address :note :log-prob])
                        (get-in trace [:trace/entries address :note :log-prob]))]
             (when (or (:trace/error proposed) (> (Math/abs (double (- joint own))) 1e-6))
               (trace/release! proposed trace)
               (throw (ex-info "A block move needs the block's complete conditional as its target: the move changed sites outside the block"
                               {:type ::incomplete-target :address address :outside (- joint own)})))
             (when-not keep-old? (trace/release! trace proposed))
             (resolve proposed))
           (catch #?(:clj Throwable :cljs :default) e (reject e))))
       reject))))

(defn move
  "One block move of the site at `address`: `steps` random-walk steps of
  `scales` against its density at `beta` (nil: untempered), drawing from the
  stream keyed by `key`, then one write-back (`keep-old?` as for it).
  Resolves {:trace :accepted :steps}; the trace is unchanged when no step was
  accepted."
  [trace address {:keys [beta scales steps key keep-old?]}]
  (fn [resolve reject]
    (try
      (let [{q0 :value {d :dist} :note} (get-in trace [:trace/entries address])
            {:keys [q accepted]} (random/in-world-stream (:trace/world trace) [::move address key]
                                                         #(walk (density d beta) q0 scales steps))]
        (if (zero? accepted)
          (resolve {:trace trace :accepted 0 :steps steps})
          ((write-back trace address q beta keep-old?)
           (fn [t] (resolve {:trace t :accepted accepted :steps steps}))
           reject)))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

;; -----------------------------------------------------------------------------
;; Adaptation: diagonal adaptive Metropolis during burn-in
;; -----------------------------------------------------------------------------

(def ^:private target-acceptance 0.234)

(defn- adapt
  "The adaptation state after an iteration that visited `q` and accepted
  `rate` of its steps: running moments of q (Welford) and a log overall scale
  moved toward the target acceptance with step 1/(i+1)^0.6."
  [{:keys [n mean m2 log-lambda] :as st} iteration q rate]
  (let [n' (inc n)
        delta (mapv - q mean)
        mean' (mapv (fn [mu d] (+ mu (/ d n'))) mean delta)
        m2' (mapv (fn [s d x mu'] (+ s (* d (- x mu')))) m2 delta q mean')]
    (assoc st :n n' :mean mean' :m2 m2'
           :log-lambda (+ log-lambda (/ (- rate target-acceptance)
                                        (Math/pow (inc iteration) 0.6))))))

(defn- scales
  "Per-coordinate step sizes: the overall scale times each coordinate's
  sample standard deviation once there are enough draws, else the initial
  step size."
  [{:keys [n m2 log-lambda step-size]}]
  (let [lambda (Math/exp log-lambda)]
    (mapv (fn [s] (if (> n 10)
                    (* lambda (Math/sqrt (max (/ s (dec n)) 1e-12)))
                    (* lambda step-size)))
          m2)))

(defn within-gibbs
  "A step for `foerster.trace/mh-chain` (`:step`): a block move of every block
  site, then one single-site random-walk MH move of a latent that is not a
  block site, if there is one. Per-coordinate scales adapt during the first
  `:burn` iterations and are fixed after. `opts`: `:step-size` (the initial
  and single-site scale), `:burn`, `:steps` (random-walk steps per block
  move; default the block's dimension)."
  [{:keys [step-size burn steps] :or {burn 0}}]
  (let [adaptation (atom {})
        propose (itrace/random-walk-proposal step-size)]
    (fn [trace {:keys [iteration] :as chain-opts}]
      (fn [resolve reject]
        (let [blocks (filterv #(block-site? trace %) (itrace/latent-addresses trace))]
          (letfn [(others [t] (vec (remove #(block-site? t %) (itrace/latent-addresses t))))
                  (go [t [address & more] moves accepted]
                      (if address
                        (let [dim (count (get-in t [:trace/entries address :value]))
                              st (or (get @adaptation address)
                                     {:n 0 :mean (vec (repeat dim 0.0)) :m2 (vec (repeat dim 0.0))
                                      :log-lambda (Math/log (/ 2.38 (Math/sqrt dim)))
                                      :step-size step-size})
                              k (or steps dim)]
                          ((move t address {:scales (scales st) :steps k :key [iteration]})
                           (fn [{t' :trace n-accepted :accepted}]
                             (when (< iteration burn)
                               (swap! adaptation assoc address
                                      (adapt st iteration (get-in t' [:trace/entries address :value])
                                             (/ (double n-accepted) k))))
                             (when-not (contains? @adaptation address)
                               (swap! adaptation assoc address st))
                             (go t' more (inc moves) (+ accepted (if (pos? n-accepted) 1 0))))
                           reject))
                        (let [done (fn [t moves accepted]
                                     (resolve {:trace t :accepted? (pos? accepted) :moves moves
                                               :accepted-moves accepted}))]
                          (if (seq (others t))
                            ((itrace/mh-step t (cond-> (assoc chain-opts :propose propose)
                                                 (seq blocks)
                                                 (assoc :select (fn [t' _]
                                                                  {:targets #{(m/pick-uniformly (others t'))}
                                                                   :log-selection #(- (Math/log (double (count (others %)))))}))))
                             (fn [r] (done (:trace r) (inc moves) (+ accepted (if (:accepted? r) 1 0))))
                             reject)
                            (done t moves accepted)))))]
            (go trace blocks 0 0)))))))
