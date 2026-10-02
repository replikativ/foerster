(ns org.replikativ.foerster.enumerate-test
  "Exact enumeration against hand-computed posteriors and evidence."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample factor]]
            [org.replikativ.foerster.enumerate :as enumerate]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- burglary
  "Pearl's alarm network: burglary and earthquake set off the alarm, which
  John reports."
  []
  (spin
   (let [burglary (sample (dist/flip 0.01) :id :burglary)
         quake (sample (dist/flip 0.02) :id :quake)
         alarm (sample (dist/flip (cond (and burglary quake) 0.95 burglary 0.94 quake 0.29 :else 0.001)) :id :alarm)]
     (observe (dist/flip (if alarm 0.9 0.05)) true :id :john)
     burglary)))

(def ^:private burglary-exact
  ;; Σ over the 8 joint states
  (let [states (for [b [true false] e [true false] a [true false]]
                 (let [pa (cond (and b e) 0.95 b 0.94 e 0.29 :else 0.001)]
                   [b (* (if b 0.01 0.99) (if e 0.02 0.98) (if a pa (- 1 pa)) (if a 0.9 0.05))]))
        z (reduce + (map second states))]
    {:p (/ (reduce + (map second (filter first states))) z) :log-z (Math/log z)}))

(deftest enumeration-is-exact
  (let [measure (b/run-infer 91 #(infer/infer (burglary) {:method :enumerate}))
        p (reduce + (map (fn [[s w]] (if (m/get-value s) w 0.0))
                         (map vector (map first (m/get-particles measure))
                              (m/normalize-log-weights (mapv second (m/get-particles measure))))))]
    (is (= 8 (count (m/get-particles measure))) "every joint state once")
    (is (< (Math/abs (- p (:p burglary-exact))) 1e-12) (str p))
    (is (< (Math/abs (- (m/log-marginal measure) (:log-z burglary-exact))) 1e-12))))

(deftest branches-depend-on-earlier-choices
  ;; k ~ uniform{1,2,3}; x ~ binomial(k, ½); observe x = 1 through a factor
  (let [measure (b/run-infer 92 #(infer/infer (spin (let [k (sample (dist/uniform-discrete 1 4) :id :k)
                                                          x (sample (dist/binomial k 0.5) :id :x)]
                                                      (factor (if (= x 1) 0.0 ##-Inf))
                                                      k))
                                              {:method :enumerate}))
        ;; P(x = 1 | k) = k / 2^k: 1/2, 1/2, 3/8
        w {1 0.5 2 0.5 3 0.375}
        z (reduce + (vals w))
        post (reduce (fn [acc [s wt]] (update acc (m/get-value s) (fnil + 0.0) wt))
                     {} (map vector (map first (m/get-particles measure))
                             (m/normalize-log-weights (mapv second (m/get-particles measure)))))]
    (doseq [k [1 2 3]]
      (is (< (Math/abs (- (get post k) (/ (w k) z))) 1e-12) (str k)))
    (is (< (Math/abs (- (m/log-marginal measure) (Math/log (/ z 3)))) 1e-12))))

(deftest continuous-sites-are-refused
  (is (= ::enumerate/infinite-support
         (try (b/run-infer 93 #(infer/infer (spin (sample (dist/normal 0.0 1.0) :id :x)) {:method :enumerate}))
              nil
              (catch clojure.lang.ExceptionInfo e
                (:type (ex-data (or (ex-cause e) e))))))))
