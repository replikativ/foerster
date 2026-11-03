(ns org.replikativ.foerster.enumerate-test
  "Exact enumeration against hand-computed posteriors and evidence."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample factor intervene!]]
            [org.replikativ.foerster.inference.enumerate :as enumerate]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.runtime.trace :as itrace]
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

(deftest impossible-branches-are-dropped
  ;; the only possible outcome is :ok; the branch with the continuous site
  ;; has probability zero and must not be run on
  (let [measure (b/run-infer 1 #(infer/infer (spin (if (sample (dist/flip 0.0) :id :z)
                                                     (sample (dist/normal 0.0 1.0) :id :x)
                                                     :ok))
                                             {:method :enumerate}))]
    (is (= [:ok] (mapv (comp m/get-value first) (m/get-particles measure))))))

(deftest nil-is-an-observable-outcome
  (let [measure (b/run-infer 2 #(infer/infer (spin (observe (dist/categorical {nil 1.0 :a 3.0}) nil :id :y) :done)
                                             {:method :enumerate}))]
    (is (< (Math/abs (- (m/log-marginal measure) (Math/log 0.25))) 1e-12))))

;; -----------------------------------------------------------------------------
;; The caller's policy decides every site
;; -----------------------------------------------------------------------------

(defn- noisy-flip
  "x ~ flip(0.3); observe flip(x ? 0.9 : 0.1) = true. With `world-do`, the
  model first intervenes on x in its own world (`intervene!`)."
  [world-do]
  (spin
   (when (some? world-do) (intervene! :x world-do))
   (let [x (sample (dist/flip 0.3) :id :x)]
     (observe (dist/flip (if x 0.9 0.1)) true :id :y)
     x)))

(defn- evidence-and-p-true [measure]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))]
    [(Math/exp (m/log-marginal measure))
     (reduce + (map (fn [[s _] w] (if (m/get-value s) w 0.0)) ps ws))]))

(deftest the-caller-s-policy-decides-latent-sites
  ;; [label world-do opts evidence P(x | y) branches]
  (doseq [[label world-do opts z p-true branches]
          [["plain" nil {} 0.34 (/ 0.27 0.34) 2]
           ;; a constraint fixes x and scores it: p(x = true) · p(y | x)
           ["constraint" nil {:policy (itrace/policy {:constraints {:x true}})} 0.27 1.0 1]
           ;; do(x = true) fixes x and scores nothing
           ["policy intervention" nil {:policy (itrace/policy {:interventions {:x true}})} 0.9 1.0 1]
           ["world intervention" true {} 0.9 1.0 1]
           ["world intervention, false" false {} 0.1 0.0 1]
           ;; a new mechanism is summed over: 0.8·0.9 + 0.2·0.1
           ["mechanism replacement" nil {:policy (itrace/policy {:interventions {:x {:dist (dist/flip 0.8)}}})}
            0.74 (/ 0.72 0.74) 2]
           ;; the temperature stays: 0.3·0.9^½ + 0.7·0.1^½
           ["temperature" nil {:policy (itrace/policy {:temperature 0.5})}
            (+ (* 0.3 (Math/sqrt 0.9)) (* 0.7 (Math/sqrt 0.1)))
            (/ (* 0.3 (Math/sqrt 0.9)) (+ (* 0.3 (Math/sqrt 0.9)) (* 0.7 (Math/sqrt 0.1)))) 2]]]
    (let [exact (b/run-infer 3 #(infer/infer (noisy-flip world-do) (assoc opts :method :enumerate)))
          [ez ep] (evidence-and-p-true exact)
          ;; importance sampling under the same policy agrees
          [iz ip] (evidence-and-p-true
                   (b/run-infer 4 #(infer/infer (noisy-flip world-do)
                                                (assoc opts :method :importance :particles 4000 :seed 4))))]
      (is (< (Math/abs (- ez z)) 1e-12) (str label ": evidence " ez))
      (is (< (Math/abs (- ep p-true)) 1e-12) (str label ": posterior " ep))
      (is (= branches (count (m/get-particles exact))) (str label ": branches"))
      (is (< (Math/abs (- iz z)) 0.03) (str label ": importance sampling's evidence " iz))
      (is (< (Math/abs (- ip p-true)) 0.03) (str label ": importance sampling's posterior " ip)))))

(deftest unsummable-policies-are-refused
  (doseq [[label opts] [["opaque policy" {:policy (fn [_sp _old] nil)}]
                        ["fresh mechanism noise"
                         {:policy (itrace/policy {:interventions {:x {:dist (dist/flip 0.8) :coupling :fresh}}})}]
                        ["unknown option" {:particles-per-branch 3}]
                        ["predictive policy" {:policy (itrace/policy {:simulate-observed? true})}]]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (b/run-infer 5 #(infer/infer (noisy-flip nil) (assoc opts :method :enumerate))))
        label)))
