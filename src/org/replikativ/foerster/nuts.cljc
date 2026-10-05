(ns org.replikativ.foerster.nuts
  "The No-U-Turn sampler on block sites (`foerster.block`), within Gibbs, as
  Stan implements it: multinomial sampling over the trajectory, the
  generalized no-U-turn criterion with the checks across merged subtrees
  (Hoffman & Gelman 2014; Betancourt 2017), divergences when the energy error
  exceeds 1000, a diagonal metric, and during warm-up dual-averaging step
  size adaptation and windowed variance estimation of the metric (Stan's
  75/25/50 windows, scaled to short warm-ups).

  NUTS samples the block's own density, so the block's target must be the
  complete conditional of its latents (`:block/target
  :complete-conditional`): a replay that changes the log probability of
  anything outside the block is refused (`::incomplete-target`) — HMC
  (`foerster.hmc`), whose fixed trajectory allows a correction on the full
  trace, handles incomplete targets.

  The kernel is `(k/nuts-kernel iterations {:burn … :target-accept 0.8
  :max-depth 10})`: the first `:burn` iterations adapt and are dropped."
  (:require [org.replikativ.foerster.block :as block]
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
  "A phase-space point: position, momentum, log density and gradient."
  [dist inv-metric q p]
  (let [[lp g] (block/value+grad! dist q)]
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

(defn transition
  "One NUTS transition of `dist` (a block distribution) from position `q0`
  with step size `eps` and diagonal inverse metric `inv-metric`, drawing from
  the current generator. Returns {:q :accept-stat :depth :n-leapfrog
  :divergent?}."
  [dist q0 eps inv-metric max-depth]
  (let [inv-metric (double-array inv-metric)
        q0 (double-array q0)
        p0 (let [n (alength inv-metric) out (double-array n)]
             (dotimes [i n]
               (aset out i (* (double (dist/draw (dist/normal 0.0 1.0))) (Math/sqrt (/ 1.0 (aget inv-metric i))))))
             out)
        z0 (point dist inv-metric q0 p0)
        h0 (hamiltonian z0)]
    (loop [depth 0 fwd z0 bck z0 sample z0 log-w 0.0 rho (:p z0) n 0 metro 0.0 divergent? false]
      (if (>= depth max-depth)
        {:q (vec (:q sample)) :accept-stat (/ metro (max 1 n)) :depth depth :n-leapfrog n :divergent? divergent?}
        (let [forward? (> (random/uniform01) 0.5)
              from (if forward? fwd bck)
              sub (build-tree dist inv-metric eps (if forward? 1 -1) depth from h0)
              n (+ n (:n sub)) metro (+ metro (:metro sub))]
          (if-not (:valid? sub)
            {:q (vec (:q sample)) :accept-stat (/ metro (max 1 n)) :depth (inc depth) :n-leapfrog n
             :divergent? (boolean (:divergent? sub))}
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
                {:q (vec (:q sample)) :accept-stat (/ metro (max 1 n)) :depth (inc depth) :n-leapfrog n
                 :divergent? divergent?}))))))))

;; =============================================================================
;; Adaptation (Stan's defaults)
;; =============================================================================

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

(defn- adapt
  "The adaptation state after warm-up iteration `i`, which ended at position
  `q` with accept statistic `stat`: the step size by dual averaging; within a
  window the positions are collected, and at its end they give the metric
  and the step size restarts."
  [{:keys [da windows draws] :as st} i q stat target]
  (let [[da eps] (da-update da stat target)
        st (assoc st :da da :eps eps)
        window (first (filter (fn [[a b]] (and (<= a i) (< i b))) windows))]
    (cond
      (nil? window) st
      (= (inc i) (second window))
      (assoc st :inv-metric (regularized-variance (conj draws q)) :draws [] :da (da-init eps))
      :else (assoc st :draws (conj draws q)))))

;; =============================================================================
;; Within Gibbs
;; =============================================================================

(defn- block-site? [trace address]
  (block/block-dist? (get-in trace [:trace/entries address :note :dist])))

(defn- nuts-move
  "Move the block site at `address` by one NUTS transition and replay; the
  chain's adaptation state is in the atom `adaptation`."
  [trace address iteration {:keys [burn target-accept max-depth constraints]} adaptation]
  (fn [resolve reject]
    (try
      (let [{q0 :value {dist :dist} :note} (get-in trace [:trace/entries address])
            dim (count q0)
            st (or (get @adaptation address)
                   {:eps 0.1 :da (da-init 0.1) :inv-metric (vec (repeat dim 1.0))
                    :windows (windows burn) :draws []})
            {:keys [q accept-stat divergent?] :as tr}
            (random/in-world-stream (:trace/world trace) [::transition address iteration]
                                    #(transition dist q0 (:eps st) (:inv-metric st) max-depth))
            st' (if (< iteration burn)
                  (cond-> (adapt st iteration q accept-stat target-accept)
                    ;; warm-up over: the averaged step size
                    (= (inc iteration) burn) (as-> s (assoc s :eps (Math/exp (:x-bar (:da s))))))
                  st)]
        (swap! adaptation assoc address st')
        (if (= q q0)
          (resolve {:trace trace :accepted? false :divergent? divergent? :stat tr})
          (let [move (itrace/policy {:keep? true :constraints constraints
                                     :draw (fn [sp _] (when (= address (:savepoint/address sp))
                                                        {:value q :symmetric? true}))})]
            ((trace/replay trace address move {:anchor? itrace/anchor?})
             (fn [proposed]
               (try
                 (let [joint (- (itrace/log-joint proposed) (itrace/log-joint trace))
                       own (- (get-in proposed [:trace/entries address :note :log-prob])
                              (get-in trace [:trace/entries address :note :log-prob]))]
                   (when (or (:trace/error proposed) (> (Math/abs (double (- joint own))) 1e-6))
                     (trace/release! proposed trace)
                     (throw (ex-info "NUTS needs the block's complete conditional as its target: the move changed sites outside the block"
                                     {:type ::incomplete-target :address address :outside (- joint own)})))
                   (trace/release! trace proposed)
                   (resolve {:trace proposed :accepted? true :divergent? divergent? :stat tr}))
                 (catch #?(:clj Throwable :cljs :default) e (reject e))))
             reject))))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

(defn within-gibbs
  "A step for `foerster.trace/mh-chain` (`:step`): a NUTS move of every block
  site, then one single-site MH move of a latent that is not a block site, if
  there is one. A fresh adaptation state per call (per chain). `opts`:
  `:burn`, `:target-accept` (0.8), `:max-depth` (10)."
  [{:keys [burn target-accept max-depth] :or {burn 0 target-accept 0.8 max-depth 10}}]
  (let [adaptation (atom {})
        divergences (atom 0)]
    (fn [trace {:keys [iteration] :as chain-opts}]
      (fn [resolve reject]
        (let [opts (assoc chain-opts :burn burn :target-accept target-accept :max-depth max-depth)
              blocks (filterv #(block-site? trace %) (itrace/latent-addresses trace))]
          (letfn [(others [t] (vec (remove #(block-site? t %) (itrace/latent-addresses t))))
                  (go [t [address & more] moves]
                      (if address
                        ((nuts-move t address iteration opts adaptation)
                         (fn [r]
                           (when (and (:divergent? r) (>= iteration burn)) (swap! divergences inc))
                           (go (:trace r) more (inc moves)))
                         reject)
                        (let [done (fn [t moves accepted]
                                     (resolve {:trace t :accepted? (pos? accepted) :moves moves
                                               :accepted-moves accepted :divergences @divergences}))]
                          (if (seq (others t))
                            ((itrace/mh-step t (assoc chain-opts
                                                      :select (fn [t' _]
                                                                {:targets #{(m/pick-uniformly (others t'))}
                                                                 :log-selection #(- (Math/log (double (count (others %)))))})))
                             (fn [r] (done (:trace r) (inc moves) (+ moves (if (:accepted? r) 1 0))))
                             reject)
                            (done t moves moves)))))]
            (go trace blocks 0)))))))
