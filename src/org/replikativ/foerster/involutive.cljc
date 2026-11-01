(ns org.replikativ.foerster.involutive
  "Involutive MCMC over savepoint traces (Neklyudov et al. 2020; Gen's
  `involutive_mh`).

  A move draws auxiliary variables u ~ q(· ; x) from the current choices x,
  maps (x, u) through an involution h to (x', u'), replays the program with
  x', and accepts with

    log α = log p(x') − log p(x) + log q(u' ; x') − log q(u ; x) + log |det ∂h/∂(x,u)|

  Random-walk, scale, swap and data-driven moves are all such pairs (q, h).
  The Jacobian term is the caller's (`:log-jacobian` from the involution);
  `fd-log-jacobian` computes it numerically for real-valued maps — raster's AD
  is the intended replacement.

  Moves may change the set of sites (reversible jump; Green 1995): the
  involution sets the sites of the new dimension through `:choices` (new
  addresses included) and names under `:removed` the sites whose values it
  moved into the auxiliary variables — those are part of the state, not
  dropped. A site the replay reaches afresh without the involution setting
  it is drawn from its prior, and one it no longer reaches and that is not
  `:removed` is dropped, both scored as in single-site MH."
  (:require [org.replikativ.spindel.trace :as trace]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.engine.protocols :as rtp]))

(defn- entry-map [t]
  (into {} (map (juxt :address identity)) (itrace/entries t)))

(defn- replay-terms
  "log q(old | new) − log q(new | old) for the sites outside the involution:
  Σ, over the latents of `old` the replay dropped (and the involution did not
  move into its auxiliaries, `removed`), of the density a reverse replay
  would draw them with, minus Σ, over the latents `new` drew afresh, of the
  density they were drawn with (`:log-proposal`: the site's prior, its
  `:proposal` or a block's :sample). −∞ when the replay redrew a value the
  reverse move would keep (`trace/irreversible-redraw?`)."
  [old new removed]
  (let [old-by (entry-map old)
        new-by (entry-map new)
        new-entries (itrace/entries new)
        fresh (filter (fn [e] (and (itrace/latent? e)
                                   (not (identical? (:note e) (:note (get old-by (:address e)))))
                                   (not (:kept? (:note e)))
                                   (not (:symmetric? (:note e)))))
                      new-entries)
        stale (filter (fn [e] (and (itrace/latent? e)
                                   (not (contains? new-by (:address e)))
                                   (not (contains? removed (:address e)))))
                      (itrace/entries old))]
    (if (itrace/irreversible-redraw? old-by new-entries)
      ##-Inf
      (- (reduce + 0.0 (map (fn [{:keys [note value]}] (itrace/redraw-log-density note value)) stale))
         (reduce + 0.0 (map (comp :log-proposal :note) fresh))))))

(defn- same-value? [a b]
  (cond (and (number? a) (number? b))
        (<= (Math/abs (- (double a) (double b)))
            (* 1e-9 (max 1.0 (Math/abs (double a)) (Math/abs (double b)))))
        (and (map? a) (map? b))
        (and (= (set (keys a)) (set (keys b))) (every? #(same-value? (get a %) (get b %)) (keys a)))
        (and (sequential? a) (sequential? b))
        (and (= (count a) (count b)) (every? true? (map same-value? a b)))
        :else (= a b)))

(defn- check-involution!
  "Apply `involution` to the new state (x', u') and require it to give back
  the old one: every address it sets must take its value in `x`, and the
  auxiliaries must come back as `u` (Gen's `check=true`)."
  [involution x u x' u']
  (let [{x'' :choices u'' :aux} (involution x' u')
        wrong (into {} (remove (fn [[a v]] (and (contains? x a) (same-value? v (get x a))))) x'')]
    (when (or (seq wrong) (not (same-value? u u'')))
      (throw (ex-info "The move is not an involution: applying it twice does not return the state"
                      {:type ::not-an-involution
                       :choices wrong
                       :aux {:expected u :actual u''}})))))

(defn step
  "One involutive MH move on `trace`.

    :propose    (fn [choices]) -> {:aux u :log-q log q(u ; x)}
    :log-q      (fn [choices u]) -> log q(u ; x), for the reverse move
    :involution (fn [choices u]) -> {:choices x' :aux u' :log-jacobian l
                                     :removed #{address}}
                where x' gives new values for some addresses of `choices`
                (or for addresses of a new dimension), and :removed names
                the sites of the old dimension it moved into u'
                (reversible jump; default none)
    :check?     apply the involution to the new state as well and require it
                to give back the old one (Gen's `check=true`); a debugging
                aid that costs one more involution per move

  `choices` is `foerster.trace/choices` of the trace ({address value}).
  Resolves {:trace t :accepted? b :log-ratio r}; the loser's worlds are
  given back."
  [trace {:keys [propose log-q involution check?]}]
  (fn [resolve reject]
    (try
      (let [x (itrace/choices trace)
            world (:trace/world trace)
            ;; every move from this state draws its own auxiliary variable
            move (rtp/swap-state! world [:inference ::moves] (fnil inc 0))
            {u :aux lq-fwd :log-q} (random/in-world-stream world [::aux move] #(propose x))
            {x' :choices u' :aux lj :log-jacobian removed :removed} (involution x u)
            changed (into {} (filter (fn [[a v]] (or (not (contains? x a)) (not= v (get x a))))) x')
            from (trace/earliest trace (keys changed))]
        (if-not from
          (resolve {:trace trace :accepted? false :log-ratio 0.0})
          ((trace/replay trace from
                         (itrace/policy {:keep? true
                                         :draw (fn [sp _]
                                                 (let [a (:savepoint/address sp)]
                                                   (when (contains? changed a)
                                                     {:value (get changed a) :symmetric? true})))})
                         {:anchor? itrace/anchor?})
           (fn [t']
             (try
               (when (and check? (not (:trace/error t')))
                 (try (check-involution! involution x u (itrace/choices t') u')
                      (catch #?(:clj Throwable :cljs :default) e
                        (trace/release! t' trace)
                        (throw e))))
               (let [ratio (if (:trace/error t')
                             ##-Inf
                             (+ (- (itrace/log-joint t') (itrace/log-joint trace))
                                (replay-terms trace t' (set removed))
                                (- (log-q (itrace/choices t') u') lq-fwd)
                                (or lj 0.0)))
                     accept? (and (not (#?(:clj Double/isNaN :cljs js/isNaN) ratio))
                                  (or (>= ratio 0.0)
                                      (< (Math/log (random/in-world-stream (:trace/world t') ::accept
                                                                           m/uniform01))
                                         ratio)))]
                 (if accept? (trace/release! trace t') (trace/release! t' trace))
                 (resolve {:trace (if accept? t' trace) :accepted? accept? :log-ratio ratio}))
               (catch #?(:clj Throwable :cljs :default) e (reject e))))
           reject)))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

;; =============================================================================
;; Numerical Jacobian
;; =============================================================================

(defn- log-abs-det
  "log |det A| by Gaussian elimination with partial pivoting."
  [rows]
  (let [n (count rows)]
    (loop [a (mapv vec rows) col 0 acc 0.0]
      (if (= col n)
        acc
        (let [pivot (apply max-key #(Math/abs (double (get-in a [% col]))) (range col n))
              a (if (= pivot col) a (assoc a col (a pivot) pivot (a col)))
              p (double (get-in a [col col]))]
          (if (zero? p)
            ##-Inf
            (recur (reduce (fn [a r]
                             (let [f (/ (get-in a [r col]) p)]
                               (assoc a r (mapv - (a r) (mapv #(* f %) (a col))))))
                           a (range (inc col) n))
                   (inc col)
                   (+ acc (Math/log (Math/abs p))))))))))

(defn fd-log-jacobian
  "log |det ∂f/∂v| at `v` by central differences, for `f` from vectors of
  reals to vectors of the same length."
  ([f v] (fd-log-jacobian f v 1e-6))
  ([f v eps]
   (let [n (count v)
         columns (for [j (range n)]
                   (let [h (* eps (max 1.0 (Math/abs (double (nth v j)))))
                         plus (f (update v j + h))
                         minus (f (update v j - h))]
                     (mapv #(/ (- %1 %2) (* 2 h)) plus minus)))]
     (log-abs-det (apply mapv vector columns)))))
