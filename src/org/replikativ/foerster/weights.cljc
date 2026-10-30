(ns org.replikativ.foerster.weights
  "Shared population mass accounting. Normalized selection weights and total
  mass are separate; neither asserts posterior accuracy or unbiased evidence.
  A zero-mass population has no posterior and cannot be drawn or resampled.")

(defn log-sum-exp [xs]
  (when-not (seq xs)
    (throw (ex-info "log-sum-exp requires non-empty sequence" {:type ::empty-population})))
  (when (some #(or (not (number? %)) (#?(:clj Double/isNaN :cljs js/isNaN) %)) xs)
    (throw (ex-info "A log weight is NaN or not numeric" {:type :org.replikativ.foerster.measure/nan-weight})))
  (let [top (reduce max ##-Inf xs)]
    (if (or (= top ##-Inf) (= top ##Inf))
      top
      (+ top (Math/log (reduce + (map #(Math/exp (- % top)) xs)))))))

(defn log-mean-exp [xs]
  (- (log-sum-exp xs) (Math/log (count xs))))

(defn normalize
  "Linear selection weights. Zero or infinite total mass requires an explicit
  :zero/:infinite :uniform policy (legacy selection inside existing engines)."
  ([xs] (normalize xs {}))
  ([xs {:keys [zero infinite]}]
   (let [z (log-sum-exp xs)]
     (cond
       (= z ##-Inf)
       (if (= :uniform zero)
         (vec (repeat (count xs) (/ 1.0 (count xs))))
         (throw (ex-info "A zero-mass population has no selection distribution" {:type ::zero-mass})))
       (= z ##Inf)
       (if (= :uniform infinite)
         (let [n (count (filter #(= ##Inf %) xs))]
           (mapv #(if (= % ##Inf) (/ 1.0 n) 0.0) xs))
         (throw (ex-info "Infinite population mass has no defined normalization" {:type ::infinite-mass})))
       :else
       ;; Subtract the maximum, then divide: preserve legacy seeded selection
       ;; rounding as well as stability for large finite scores.
       (let [top (reduce max xs)
             ws (mapv #(Math/exp (- % top)) xs)
             total (reduce + ws)]
         (mapv #(/ % total) ws))))))

(defn from-log
  "Canonical accounting from raw log weights and an accumulated log normalizer.
  log-mass is log-normalizer + logmeanexp(raw). A zero-mass state stays explicit."
  ([xs] (from-log xs 0.0))
  ([xs log-normalizer]
   (let [xs (vec xs)
         z (log-sum-exp xs)]
     (when (or (= ##Inf z) (not (number? log-normalizer))
               (#?(:clj Double/isNaN :cljs js/isNaN) log-normalizer)
               (= ##Inf log-normalizer))
       (throw (ex-info "Population mass must be finite or zero" {:type ::invalid-mass})))
     (if (or (= ##-Inf z) (= ##-Inf log-normalizer))
       {:log-weights (vec (repeat (count xs) ##-Inf)) :log-mass ##-Inf :status :zero-mass}
       {:log-weights (mapv #(- % z) xs)
        :log-mass (+ log-normalizer (- z (Math/log (count xs))))
        :status :weighted}))))

(defn uniform
  "Equal selection weights at the given total log mass."
  ([n] (uniform n 0.0))
  ([n log-mass]
   (when-not (pos-int? n)
     (throw (ex-info "A population needs positive size" {:type ::empty-population})))
   (from-log (vec (repeat n 0.0)) log-mass)))

(defn reweight [state increments]
  (when-not (= (count increments) (count (:log-weights state)))
    (throw (ex-info "One correction is required per particle" {:type ::weight-count-mismatch})))
  (let [next (mapv + (:log-weights state) increments)
        z (log-sum-exp next)]
    (when (= ##Inf z)
      (throw (ex-info "Infinite incremental population mass" {:type ::infinite-mass})))
    (if (= ##-Inf z)
      (uniform (count next) ##-Inf)
      {:log-weights (mapv #(- % z) next)
       :log-mass (+ (:log-mass state) z)
       :status :weighted})))

(defn selection [state]
  (normalize (:log-weights state)))

(defn resampled [state n]
  (when (= :zero-mass (:status state))
    (throw (ex-info "Cannot resample a zero-mass population" {:type ::zero-mass})))
  (uniform n (:log-mass state)))
