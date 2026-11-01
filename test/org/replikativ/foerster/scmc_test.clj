(ns org.replikativ.foerster.scmc-test
  "GIGA against known combinations, and sequential core-set Monte Carlo on a
  conjugate regression against its exact posterior and evidence."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.scmc :as scmc]))

(defn- combine [w ls]
  (let [out (double-array (alength ^doubles (first ls)))]
    (doseq [[wj ^doubles l] (map vector w ls) i (range (alength out))]
      (aset out i (+ (aget out i) (* wj (aget l i)))))
    out))

(defn- rel-error [w ls ^doubles target]
  (let [a (combine w ls)]
    (/ (Math/sqrt (reduce + (map #(let [e (- %1 %2)] (* e e)) a target)))
       (Math/sqrt (reduce + (map #(* % %) target))))))

(deftest giga-recovers-a-sparse-combination
  ;; target = 3·e₀ + 2·e₂ over four orthogonal vectors of different norms
  (let [ls [(double-array [2 0 0 0]) (double-array [0 1 0 0]) (double-array [0 0 5 0]) (double-array [0 0 0 3])]
        target (double-array [6 0 10 0])
        w (scmc/giga ls target 4)]
    (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-9) w [3.0 0.0 2.0 0.0])))))

(deftest giga-is-sparse-and-improves-with-its-budget
  (let [rng (java.util.Random. 1)
        ls (vec (repeatedly 100 (fn [] (double-array (repeatedly 60 #(.nextGaussian rng))))))
        target (combine (repeat 100 1.0) ls)
        errors (for [m [5 20 60]]
                 (let [w (scmc/giga ls target m)]
                   (is (<= (count (filter pos? w)) m))
                   (is (every? #(>= % 0.0) w))
                   (rel-error w ls target)))]
    (is (apply > errors)))
  (testing "a zero target"
    (is (= [0.0 0.0] (scmc/giga [(double-array [1 0]) (double-array [0 1])] (double-array [0 0]) 2)))))

;; NNLS and the refit after GIGA, against SciPy's `nnls` and the reference
;; implementation's `GIGA.build(m); GIGA.optimize()` (bayesian-coresets as
;; vendored by ~/Development/scmc), whose outputs are recorded here.

(deftest nnls-matches-scipy
  (testing "SciPy's documented example"
    (let [x (scmc/nnls [(double-array [1 1 0]) (double-array [0 0 1])] (double-array [2 1 1]))]
      (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12) x [1.5 1.0])))))
  (testing "the unconstrained solution is negative: the constraint is active"
    ;; least squares gives (-1, 1); NNLS gives (0, 1/2), residual 1/√2
    (let [ls [(double-array [1 0 0]) (double-array [1 1 0])]
          target (double-array [0 1 0])
          x (scmc/nnls ls target)]
      (is (= 0.0 (first x)))
      (is (< (Math/abs (- 0.5 (second x))) 1e-12))
      (is (< (Math/abs (- (Math/sqrt 0.5) (rel-error x ls target))) 1e-12))))
  (testing "a target outside the cone"
    (is (= [0.0 0.0] (scmc/nnls [(double-array [1 1 0]) (double-array [0 0 1])] (double-array [-1 -1 -1])))))
  (testing "dependent vectors"
    (let [x (scmc/nnls [(double-array [1 0]) (double-array [2 0]) (double-array [0 1])] (double-array [2 3]))]
      (is (< (rel-error x [(double-array [1 0]) (double-array [2 0]) (double-array [0 1])] (double-array [2 3])) 1e-12))
      (is (every? #(>= % 0.0) x)))))

(defn- logistic-columns
  "Centred, √π-scaled log-likelihood columns of 120 binomial-logistic windows
  at 150 particles around (-2, 1.8), as SCMC's projection builds them, and
  their sum: GIGA's vectors are nearly collinear here."
  []
  (let [rng (java.util.Random. 5)
        sig (fn [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))
        rows (vec (repeatedly 120 (fn [] (let [x (- (* 2.0 (.nextDouble rng)) 1.0)
                                               p (sig (+ -2.0 (* 2.0 x)))]
                                           [x (count (filter #(< % p) (repeatedly 8 #(.nextDouble rng))))]))))
        qs (vec (repeatedly 150 #(vector (+ -2.0 (* 0.15 (.nextGaussian rng))) (+ 1.8 (* 0.3 (.nextGaussian rng))))))
        ll (fn [[a b] [x k]] (let [eta (+ a (* b x))] (- (* k eta) (* 8.0 (Math/log1p (Math/exp eta))))))
        r (Math/sqrt (/ 1.0 150))
        cols (mapv (fn [u] (let [f (mapv #(ll % u) qs) mu (/ (reduce + f) 150)]
                             (double-array (map #(* r (- % mu)) f)))) rows)]
    {:cols cols :target (combine (repeat 120 1.0) cols)}))

(defn- abs-error [w ls ^doubles target]
  (Math/sqrt (reduce + (map #(let [e (- %1 %2)] (* e e)) (combine w ls) target))))

(deftest the-refit-matches-the-reference-implementation
  (testing "a 3-vector instance: GIGA stops short, NNLS on its support is exact"
    (let [ls [(double-array [1 0 0]) (double-array [0 1 0]) (double-array [1 1 0.5])]
          target (double-array [1 2 0.2])
          g (scmc/giga ls target 3)
          r (scmc/nnls-refit ls target g)]
      (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12) g [0.14215948111407847 1.0917970240366273 0.857840518885921])))
      (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12) r [0.6 1.6 0.4])))
      (is (< (abs-error r ls target) 1e-12))
      (testing "GIGA's support of two is already optimal"
        (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12)
                               (scmc/nnls-refit ls target (scmc/giga ls target 2)) [0.0 1.12 0.88]))))))
  (testing "nearly collinear log-likelihood vectors, budget 16"
    (let [{:keys [cols target]} (logistic-columns)
          g (scmc/giga cols target 16)
          r (scmc/nnls-refit cols target g)
          support (fn [w] (into {} (keep-indexed (fn [j x] (when (pos? x) [j x]))) w))
          close? (fn [expected actual]
                   (and (= (set (keys expected)) (set (keys actual)))
                        (every? (fn [[j x]] (< (Math/abs (- x (actual j))) (* 1e-9 (max 1.0 x)))) expected)))]
      (is (close? {23 6.618675887571942 33 1.4814124435848632 39 1.2828098605491924 50 0.5192804668750788
                   72 2.444701451076503 74 38.79253944804988 91 0.7273845404560357 108 54.38239758722264
                   119 0.6834538524209867}
                  (support g)))
      ;; the refit keeps 5 of GIGA's 9 points
      (is (close? {23 19.182551833104505 50 1.2764257005361201 74 40.25328839173929
                   108 54.20953565118226 119 5.966433493819927}
                  (support r)))
      (is (every? #(>= % 0.0) r))
      (is (< (Math/abs (- (abs-error g cols target) 0.19426509152533136)) 1e-9))
      (is (< (Math/abs (- (abs-error r cols target) 0.008607328638647355)) 1e-9))
      (is (<= (abs-error r cols target) (abs-error g cols target))))))

;; y = a + b·x + N(0, 0.5²), (a, b) ~ N(0, I): the posterior and the evidence
;; are Gaussian in closed form
(def ^:private sigma2 0.25)

(defn- regression [n seed]
  (let [rng (java.util.Random. seed)
        xs (vec (repeatedly n #(.nextGaussian rng)))]
    (mapv (fn [x] [x (+ 0.7 (* -1.3 x) (* 0.5 (.nextGaussian rng)))]) xs)))

(defn- exact [data]
  (let [n (count data) xs (map first data) ys (map second data)
        sxx (reduce + (map #(* % %) xs)) sx (reduce + xs)
        syy (reduce + (map #(* % %) ys)) sy (reduce + ys) sxy (reduce + (map (fn [[x y]] (* x y)) data))
        ;; precision I + XᵀX/σ², X = [1 x]
        p00 (+ 1.0 (/ n sigma2)) p01 (/ sx sigma2) p11 (+ 1.0 (/ sxx sigma2))
        det (- (* p00 p11) (* p01 p01))
        r0 (/ sy sigma2) r1 (/ sxy sigma2)]
    {:a (/ (- (* p11 r0) (* p01 r1)) det) :b (/ (- (* p00 r1) (* p01 r0)) det)
     :a-sd (Math/sqrt (/ p11 det)) :b-sd (Math/sqrt (/ p00 det))
     ;; Woodbury and the determinant lemma on N(y; 0, σ²I + XXᵀ)
     :log-z (- (* -0.5 n (Math/log (* 2 Math/PI sigma2)))
               (* 0.5 (Math/log det))
               (* 0.5 (- (/ syy sigma2) (/ (+ (* p11 r0 r0) (* -2.0 p01 r0 r1) (* p00 r1 r1)) det))))}))

(defn- model [data batch]
  {:prior-sample (fn [] [(dist/draw (dist/normal 0.0 1.0)) (dist/draw (dist/normal 0.0 1.0))])
   :prior-log-density (fn [[a b]] (- (* -0.5 (+ (* a a) (* b b))) (* 2.0 0.9189385332046727)))
   :log-lik (fn [[a b] [x y]] (let [r (- y a (* b x))]
                                (- (/ (* -0.5 r r) sigma2) (* 0.5 (Math/log sigma2)) 0.9189385332046727)))
   :batches (partition batch data)})

(defn- moments [qs j]
  (let [v (map #(nth % j) qs) n (count v) mu (/ (reduce + v) n)]
    [mu (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) v)) (dec n)))]))

(deftest scmc-on-a-conjugate-regression
  ;; 2000 points in 100 batches, at most 50 of them remembered
  (random/set-seed! 3)
  (let [data (regression 2000 2)
        {:keys [a b a-sd b-sd log-z]} (exact data)
        r (scmc/scmc (model data 20) {:particles 500 :memory 50 :seed [1 50]})
        [am asd] (moments (mapv (comp m/get-value first) (m/get-particles r)) 0)
        [bm bsd] (moments (mapv (comp m/get-value first) (m/get-particles r)) 1)]
    (is (< (Math/abs (/ (- am a) a-sd)) 0.15))
    (is (< (Math/abs (/ (- bm b) b-sd)) 0.15))
    (is (< (Math/abs (- 1.0 (/ asd a-sd))) 0.1))
    (is (< (Math/abs (- 1.0 (/ bsd b-sd))) 0.1))
    (is (< (Math/abs (- (m/log-marginal r) log-z)) 0.5))
    (testing "the memory bound"
      (is (<= (count (:coreset r)) 50))
      (is (<= (:max-coreset-size r) 70))
      (is (= 100 (:steps r))))))

(deftest with-memory-for-all-data-nothing-is-compressed
  (random/set-seed! 4)
  (let [data (regression 200 5)
        r (scmc/scmc (model data 20) {:particles 300 :memory 200 :seed [2]})]
    (is (= 200 (count (:coreset r))))
    (is (every? #(= 1.0 (first %)) (:coreset r)))
    (is (< (Math/abs (- (m/log-marginal r) (:log-z (exact data)))) 0.5))))

(deftest refusals
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":refit" (scmc/scmc (model [] 1) {:refit :lasso})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":memory" (scmc/scmc (model [] 1) {:memory 0})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":particles" (scmc/scmc (model [] 1) {:particles 0}))))

(deftest batches-are-consumed-as-they-come
  ;; a lazy stream of batches: each is realized when its step reaches it
  (random/set-seed! 6)
  (let [data (regression 200 7)
        realized (atom 0)
        batches (map (fn [b] (swap! realized inc) b) (partition 20 data))
        seen (atom [])
        m (model data 20)
        log-lik (:log-lik m)
        r (scmc/scmc (assoc m
                            :batches batches
                            :log-lik (fn [q d] (swap! seen conj @realized) (log-lik q d)))
                     {:particles 50 :memory 30 :seed [3]})]
    (is (= 10 @realized))
    (is (= 50 (count (mapv (comp m/get-value first) (m/get-particles r)))))
    (testing "the first batch was scored before the last one was realized"
      (is (< (first @seen) 10)))))

(deftest a-batch-no-particle-explains-is-refused
  (random/set-seed! 7)
  (let [m (assoc (model (regression 40 8) 20)
                 :log-lik (fn [_ [x _]] (if (> x 100.0) 0.0 ##-Inf)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"zero likelihood"
                          (scmc/scmc m {:particles 20 :memory 10 :seed [4]})))))

(deftest settled-result-uses-the-shared-measure-protocol
  (let [result (scmc/scmc {:prior-sample (constantly [2.0])
                           :prior-log-density (constantly 0.0)
                           :log-lik (constantly 0.0)
                           :batches []}
                          {:particles 8 :seed 17})
        particles (m/get-particles result)]
    (is (satisfies? m/PMeasure result))
    (is (= :empirical (m/measure-type result)))
    (is (= (vec (repeat 8 [2.0])) (mapv (comp m/get-value first) particles)))
    (is (every? #(= 0.0 (second %)) particles))
    (is (= 8.0 (m/effective-sample-size result)))
    (is (= 0.0 (m/log-marginal result)))
    (is (= 2.0 (:mean (m/measure-stats result (comp first m/get-value)))))
    (is (= :weighted-population (get-in result [:inference-contract :semantics])))
    (is (= [] (:coreset result)))
    (is (not (contains? result :log-z)))))

(defn- counting-model
  "The regression model with its log-likelihood calls counted in `calls`."
  [calls]
  (let [m (model [] 1)
        log-lik (:log-lik m)]
    (assoc (dissoc m :batches) :log-lik (fn [q d] (swap! calls inc) (log-lik q d)))))

(deftest without-the-refit-scmc-is-the-giga-only-projection
  ;; :refit :none reproduces the implementation before the refit existed,
  ;; bit for bit (recorded on origin/main 24d9c1a)
  (let [calls (atom 0)
        state (reduce scmc/append-batch
                      (scmc/start (counting-model calls) {:particles 50 :memory 30 :seed [5] :refit :none})
                      (partition 20 (regression 300 9)))]
    (is (= -1699575438 (hash (:particles state))))
    (is (= -231.1534641942093 (:log-normalizer state)))
    (is (= 28 (count (:coreset state))))
    (is (= 1500 (:moves state)))
    ;; 111,950 before the likelihood cache
    (is (= 68800 @calls))
    (testing "the default refits"
      (let [refit (reduce scmc/append-batch
                          (scmc/start (counting-model (atom 0)) {:particles 50 :memory 30 :seed [5]})
                          (partition 20 (regression 300 9)))]
        (is (= :nnls (:refit refit)))
        (is (not= (:particles state) (:particles refit)))
        ;; recorded with the refit, before the likelihood cache and the
        ;; allocation-free GIGA: both leave every draw unchanged
        (is (= 2013277641 (hash (:particles refit))))
        (is (= -231.1888303894612 (:log-normalizer refit)))
        (is (= 25 (count (:coreset refit))))))))

(defn- counted-steps
  "Each step's likelihood calls, the core-set size it started from and its
  :last-update; `prepare` maps the state before each step."
  [prepare opts batches]
  (let [calls (atom 0)]
    (loop [state (scmc/start (counting-model calls) opts) [b & more] batches out []]
      (if b
        (let [before (prepare state)
              _ (reset! calls 0)
              next (scmc/append-batch before b)]
          (recur next more (conj out {:calls @calls :coreset (count (:coreset state))
                                      :last-update (:last-update next) :state next})))
        out))))

(deftest each-likelihood-is-evaluated-once-per-state
  ;; N = 20 particles of dimension 2, batches of 5, memory 8. A step scores
  ;; the batch at every particle (N·b), and each move's two random-walk
  ;; proposals score the expanded core-set E (2·N·|E|). The start states'
  ;; likelihoods come from the step before and the batch's weights, and the
  ;; projection reuses the moves' values. Without the cache a step also
  ;; costs N·|E| for the start states, and N·|E| more when it compresses.
  (let [opts {:particles 20 :memory 8 :seed [9]}
        batches (partition 5 (regression 30 3))
        cached (counted-steps identity opts batches)
        uncached (counted-steps #(dissoc % :coreset-log-likelihoods) opts batches)]
    (is (some #(get-in % [:last-update :compressed?]) cached))
    (doseq [{:keys [calls coreset last-update]} cached]
      (let [e (+ coreset 5)]
        (is (= (+ (* 20 5) (* 2 20 e)) calls))
        (is (= {:reweight (* 20 5) :move (* 2 20 e) :projection 0} (:likelihood-evaluations last-update)))
        (is (= (> e 8) (:compressed? last-update)))))
    (testing "without the cache, the same steps cost what they did before it"
      (doseq [{:keys [calls coreset last-update]} uncached]
        (let [e (+ coreset 5)
              starts (* 20 e)]
          (is (= (+ (* 20 5) (* 2 20 e) starts) calls))
          (is (= (+ (* 2 20 e) starts) (get-in last-update [:likelihood-evaluations :move]))))))
    (testing "and give the same states, bit for bit"
      (is (= (mapv #(dissoc (:state %) :last-update :model :coreset-log-likelihoods) cached)
             (mapv #(dissoc (:state %) :last-update :model :coreset-log-likelihoods) uncached)))
      (is (= (mapv #(get-in (:state %) [:coreset-log-likelihoods :rows]) cached)
             (mapv #(get-in (:state %) [:coreset-log-likelihoods :rows]) uncached))))))

(deftest the-projection-inflation-is-an-opt-in-deviation
  (let [opts {:particles 30 :memory 12 :seed [12]}
        batches (partition 5 (regression 60 13))
        run (fn [opts] (reduce scmc/append-batch (scmc/start (counting-model (atom 0)) opts) batches))
        default (run opts)]
    (testing "k = 1 is the reference projection"
      (let [strip #(-> % (dissoc :model :projection-inflation) (update :coreset-log-likelihoods :rows))]
        (is (= (strip default)
               (strip (run (assoc opts :projection-inflation 1)))
               (strip (run (assoc opts :projection-inflation 1.0)))))))
    (testing "k = 3 fits the projection at spread points, at N·|E| more evaluations"
      (let [steps (counted-steps identity (assoc opts :projection-inflation 3) batches)
            inflated (:state (peek steps))]
        (is (not= (:coreset default) (:coreset inflated)))
        (is (= 3 (:projection-inflation inflated)))
        (doseq [{:keys [coreset last-update calls]} steps]
          (let [e (+ coreset 5)
                projection (if (:compressed? last-update) (* 30 e) 0)]
            (is (= projection (get-in last-update [:likelihood-evaluations :projection])))
            (is (= (+ (* 30 5) (* 2 30 e) projection) calls))))))
    (testing "a likelihood that is not finite at a spread point is refused"
      (let [m (update (counting-model (atom 0)) :log-lik
                      (fn [ll] (fn [[_ b :as q] d] (if (> (Math/abs (double b)) 5.0) ##-Inf (ll q d)))))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not finite"
                              (reduce scmc/append-batch (scmc/start m (assoc opts :projection-inflation 1e3)) batches)))))
    (testing "refusals"
      (doseq [k [0 -1 ##Inf :wide]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #":projection-inflation"
                              (scmc/start (counting-model (atom 0)) (assoc opts :projection-inflation k))))))))

(deftest nnls-agrees-with-scipy-on-recorded-cases
  ;; 58 instances (random, collinear, badly scaled, in-cone, zero columns
  ;; and target) with SciPy's `nnls` solution and residual, generated by
  ;; numpy (default_rng(1)) in the SCMC review
  (doseq [{:keys [label cols b res]} (edn/read-string (slurp (io/resource "org/replikativ/foerster/scmc_nnls_cases.edn")))]
    (let [ls (mapv double-array cols)
          target (double-array b)
          x (scmc/nnls ls target)
          scale (max 1.0 (Math/sqrt (reduce + (map #(* % %) b))))]
      (is (every? #(>= % 0.0) x) label)
      (is (<= (abs-error x ls target) (+ res (* 1e-9 scale))) label))))

(defn- moments-of [state]
  (mapv (fn [j] (/ (reduce + (map #(nth % j) (:particles state))) (count (:particles state)))) [0 1]))

(deftest a-cache-from-another-model-core-set-or-population-is-not-used
  ;; The cached likelihoods belong to one log-likelihood, core-set and
  ;; population; after any change a step must equal the step without the
  ;; cache (it re-evaluates instead)
  (let [model (dissoc (model [] 1) :batches)
        batches (partition 10 (regression 80 9))
        state (reduce scmc/append-batch (scmc/start model {:particles 40 :memory 15 :seed [3]}) (take 4 batches))
        next-batch (nth batches 4)
        same? (fn [edited]
                (let [cached (scmc/append-batch edited next-batch)
                      fresh (scmc/append-batch (dissoc edited :coreset-log-likelihoods) next-batch)]
                  (and (= (:particles cached) (:particles fresh))
                       (= (:coreset cached) (:coreset fresh))
                       (= (get-in cached [:coreset-log-likelihoods :rows])
                          (get-in fresh [:coreset-log-likelihoods :rows])))))]
    (is (true? (:compressed? state)))
    (is (same? state) "unchanged: the cache is used and changes nothing")
    (is (same? (update state :model update :log-lik (fn [ll] (fn [q d] (* 0.25 (ll q d))))))
        "another log-likelihood")
    (is (same? (assoc-in state [:coreset 0 1] [0.0 100.0])) "an edited core-set datum")
    (is (same? (update-in state [:coreset 0 0] * 3.0)) "an edited core-set weight")
    (is (same? (update state :coreset subvec 1)) "a dropped core-set point")
    (is (same? (update state :particles (fn [ps] (vec (cons (mapv + (first ps) [0.5 0.5]) (rest ps))))))
        "a moved particle")
    (is (same? (update state :particles (comp vec reverse))) "reordered particles")
    (testing "and the stale cache would have changed the posterior"
      (let [edited (assoc-in state [:coreset 0 1] [0.0 100.0])
            fresh (scmc/append-batch (dissoc edited :coreset-log-likelihoods) next-batch)
            forged (scmc/append-batch (assoc-in edited [:coreset-log-likelihoods :coreset] (:coreset edited))
                                      next-batch)]
        (is (not= (moments-of fresh) (moments-of forged)))))))
