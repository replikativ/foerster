(ns org.replikativ.foerster.nuts-test
  "NUTS with adaptation on block sites, and constrained latents, against
  exact posteriors."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.nuts :as nuts]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private scales [0.05 0.2 0.5 1.0 2.0 3.0 5.0 8.0 12.0 20.0])

(def ^:private scaled-gaussian
  ;; independent N(0, sᵢ²) over four orders of magnitude: only an adapted
  ;; metric samples it well
  (let [lp #(reduce + (map (fn [x s] (* -0.5 (/ (* x x) (* s s)))) (vec %) scales))]
    (block/block {:block/id :scaled :block/target :complete-conditional
                  :block/latents [{:name :x :shape [10] :support :real}]}
                 {:log-density (fn [th _] (lp th))
                  :value+grad (fn [th _] [(lp th) (double-array (map (fn [x s] (- (/ x (* s s)))) (vec th) scales))])})))

(deftest nuts-adapts-to-an-ill-conditioned-target
  (let [measure (b/run-infer 121 #(infer/infer (spin (sample (block/block-dist scaled-gaussian nil) :id :x
                                                             :init (vec (repeat 10 0.5))))
                                               {:method :nuts :iterations 1000 :chains 4 :burn 500}))]
    (doseq [j (range 10)]
      (let [cs (d/chains measure #(nth % j))]
        (is (< (d/rhat cs) 1.01) (str j))
        (is (< (Math/abs (- 1.0 (/ (:sd (d/summary measure #(nth % j))) (nth scales j)))) 0.1) (str j))))))

(deftest the-fisher-metric-is-exact-for-a-gaussian
  ;; with g = −q/s², √(Var q / Var g) = s² for any sample
  (let [rng (java.util.Random. 5)
        qs (vec (repeatedly 30 (fn [] (mapv #(* % (.nextGaussian rng)) scales))))
        gs (mapv (fn [q] (mapv (fn [x s] (- (/ x (* s s)))) q scales)) qs)]
    (is (every? true? (map #(< (Math/abs (- 1.0 (/ %1 (* %2 %2)))) 1e-9)
                           (@#'nuts/fisher-variance qs gs) scales)))))

(deftest nuts-adapts-a-fisher-metric
  (let [measure (b/run-infer 122 #(infer/infer (spin (sample (block/block-dist scaled-gaussian nil) :id :x
                                                             :init (vec (repeat 10 0.5))))
                                               {:method :nuts :iterations 1000 :chains 4 :burn 500 :metric :fisher}))]
    (doseq [j (range 10)]
      (let [cs (d/chains measure #(nth % j))]
        (is (< (d/rhat cs) 1.01) (str j))
        (is (< (Math/abs (- 1.0 (/ (:sd (d/summary measure #(nth % j))) (nth scales j)))) 0.1) (str j)))))
  (is (thrown? AssertionError (k/nuts-kernel 100 {:metric :dense}))))

(def ^:private ys [2.1 3.4 1.7 2.9 2.5 3.8 2.2 3.1])

(defn- normal-lp+grad
  "mu ~ N(0, 10), sigma ~ half-normal(2), ys ~ N(mu, sigma), in (mu, sigma)."
  [^doubles x]
  (let [mu (aget x 0) s (aget x 1)]
    [(+ (* -0.5 (/ (* mu mu) 100.0)) (- (* 0.5 (/ (* s s) 4.0)))
        (reduce + (map #(- (* -0.5 (/ (* (- % mu) (- % mu)) (* s s))) (Math/log s)) ys)))
     (double-array [(+ (- (/ mu 100.0)) (reduce + (map #(/ (- % mu) (* s s)) ys)))
                    (+ (- (/ s 4.0)) (reduce + (map #(- (/ (* (- % mu) (- % mu)) (* s s s)) (/ 1.0 s)) ys)))])]))

(def ^:private mean-and-scale
  (block/block {:block/id :ms :block/coordinates :constrained :block/target :complete-conditional
                :block/latents [{:name :mu :shape [] :support :real}
                                {:name :sigma :shape [] :support :positive}]}
               {:log-density (fn [x _] (first (normal-lp+grad x)))
                :value+grad (fn [x _] (normal-lp+grad x))}))

(def ^:private grid-means
  (delay
    (let [pts (for [mu (range 0.0 5.0 0.01) s (range 0.005 4.0 0.01)]
                [mu s (first (normal-lp+grad (double-array [mu s])))])
          top (apply max (map peek pts))
          ws (map #(Math/exp (- (peek %) top)) pts)
          z (reduce + ws)]
      [(/ (reduce + (map * (map first pts) ws)) z) (/ (reduce + (map * (map second pts) ws)) z)])))

(deftest constrained-latents-in-natural-coordinates
  (testing "σ > 0 by its log"
    (let [measure (b/run-infer 122 #(infer/infer (spin (sample (block/block-dist mean-and-scale nil) :id :th :init [2.0 0.0]))
                                                 {:method :nuts :iterations 1000 :chains 4 :burn 500}))
          [mu sigma] @grid-means
          s (fn [i] (d/summary measure #(nth (block/constrain mean-and-scale %) i)))]
      (is (< (Math/abs (- (:mean (s 0)) mu)) (* 4 (:mcse (s 0)))) (str (s 0) " vs " mu))
      (is (< (Math/abs (- (:mean (s 1)) sigma)) (* 4 (:mcse (s 1)))) (str (s 1) " vs " sigma))))
  (testing "p in (0, 1) by its logit: Beta(2, 2), 7 of 10 → Beta(9, 5)"
    (let [b (block/block {:block/id :p :block/coordinates :constrained :block/target :complete-conditional
                          :block/latents [{:name :p :shape [] :support [:interval 0.0 1.0]}]}
                         {:log-density (fn [x _] (let [p (aget ^doubles x 0)] (+ (* 8 (Math/log p)) (* 4 (Math/log (- 1 p))))))
                          :value+grad (fn [x _] (let [p (aget ^doubles x 0)]
                                                  [(+ (* 8 (Math/log p)) (* 4 (Math/log (- 1 p))))
                                                   (double-array [(- (/ 8 p) (/ 4 (- 1 p)))])]))})
          measure (b/run-infer 123 #(infer/infer (spin (sample (block/block-dist b nil) :id :p :init [0.0]))
                                                 {:method :nuts :iterations 1000 :chains 4 :burn 500}))
          s (d/summary measure #(first (block/constrain b %)))]
      (is (< (Math/abs (- (:mean s) (/ 9.0 14))) (* 4 (:mcse s))) (str s))
      (is (< (Math/abs (- (:sd s) (Math/sqrt (/ (* 9 5) (* 14 14 15.0))))) 0.01) (str s)))))

(deftest an-incomplete-target-is-refused
  ;; the block covers mu's prior only; the observation of mu is outside it
  (let [b (block/block {:block/id :prior-only :block/target :complete-conditional
                        :block/latents [{:name :mu :shape [] :support :real}]}
                       {:log-density (fn [th _] (* -0.5 (let [m (aget ^doubles th 0)] (* m m))))
                        :value+grad (fn [th _] (let [m (aget ^doubles th 0)] [(* -0.5 m m) (double-array [(- m)])]))})]
    (is (= ::nuts/incomplete-target
           (try (b/run-infer 124 #(infer/infer (spin (let [[mu] (sample (block/block-dist b nil) :id :mu :init [0.0])]
                                                       (observe (dist/normal mu 1.0) 2.0 :id :y)
                                                       mu))
                                               {:method :nuts :iterations 20 :chains 1 :burn 10}))
                nil
                (catch Exception e
                  (loop [e e] (cond (nil? e) nil
                                    (= ::nuts/incomplete-target (:type (ex-data e))) ::nuts/incomplete-target
                                    :else (recur (ex-cause e))))))))))

(deftest a-numeric-gradient-is-an-explicit-opt-in
  ;; the μ/σ block from its density alone, gradient by central differences
  (let [b (block/block {:block/id :ms-numeric :block/coordinates :constrained :block/target :complete-conditional
                        :block/latents [{:name :mu :shape [] :support :real}
                                        {:name :sigma :shape [] :support :positive}]}
                       (block/with-numeric-gradient {:log-density (fn [x _] (first (normal-lp+grad x)))}))
        x (double-array [2.3 -0.4])
        [_ g] ((block/capability b :value+grad) x nil)
        [_ g'] ((block/capability mean-and-scale :value+grad) x nil)]
    (is (every? #(< (Math/abs %) 1e-5) (map - g g')))))

(deftest an-interval-transform-is-exact-in-its-tails
  ;; p ~ Uniform(0, 1) in natural coordinates: in θ = logit p the density is
  ;; σ(θ)(1 − σ(θ)) ≈ e^−|θ|, also where σ rounds to 1
  (let [b (block/block {:block/id :p :block/coordinates :constrained
                        :block/latents [{:name :p :shape [] :support [:interval 0.0 1.0]}]}
                       {:log-density (fn [_ _] 0.0)
                        :value+grad (fn [_ _] [0.0 (double-array [0.0])])})
        d (block/block-dist b nil)
        [lp [g]] (block/value+grad d [40.0])]
    (is (< (Math/abs (- lp -40.0)) 1e-9) (str lp))
    (is (< (Math/abs (- g -1.0)) 1e-9) (str g))))

(deftest explicit-seeds-and-portable-sampler-statistics
  (let [target (block/block {:block/id :seeded :block/target :complete-conditional
                             :block/latents [{:name :x :shape [] :support :real}]}
                            {:log-density (fn [x _] (* -0.5 (aget x 0) (aget x 0)))
                             :value+grad (fn [x _] [(* -0.5 (aget x 0) (aget x 0))
                                                    (double-array [(- (aget x 0))])])})
        run (fn [process-seed seed]
              (b/run-infer process-seed
                           #(infer/infer (spin (sample (block/block-dist target nil) :id :x :init [0.0]))
                                         {:method :nuts :seed seed :iterations 30 :burn 10 :chains 2 :max-depth 2})))
        a (run 12 731) b (run 999 731) other (run 12 732)
        values #(mapv (comp measure/get-value first) (measure/get-particles %))
        stats (mapv (comp :sampler-stats first) (measure/get-particles a))]
    (is (= (values a) (values b)))
    (is (not= (values a) (values other)))
    (is (= 731 (:seed a)))
    (is (= [20 20] (:chain-lengths a)))
    (is (= 40 (count stats)))
    (is (every? boolean? (map :diverging stats)))
    (is (every? pos? (map :n-leapfrog stats)))
    (is (every? #(<= 1 (:tree-depth %) 2) stats))
    (is (every? #(<= 0.0 (:accept-stat %) 1.0) stats))
    (is (every? #(= (:reached-max-depth? %) (= 2 (:tree-depth %))) stats))
    (is (every? #(= 1 (:blocks %)) stats))))

(deftest divergent-transitions-remain-visible-in-samples
  (let [target (block/block {:block/id :narrow :block/target :complete-conditional
                             :block/latents [{:name :x :shape [] :support :real}]}
                            {:log-density (fn [x _] (* -5e11 (aget x 0) (aget x 0)))
                             :value+grad (fn [x _] [(* -5e11 (aget x 0) (aget x 0))
                                                    (double-array [(* -1e12 (aget x 0))])])})
        result (b/run-infer 21 #(infer/infer (spin (sample (block/block-dist target nil) :id :x :init [0.0]))
                                             {:method :nuts :seed 123 :iterations 4 :burn 0 :chains 2 :max-depth 2}))]
    (is (some (comp :diverging :sampler-stats first) (measure/get-particles result)))))
