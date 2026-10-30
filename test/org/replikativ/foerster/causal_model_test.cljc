(ns org.replikativ.foerster.causal-model-test
  (:require #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is async]])
            [org.replikativ.foerster.causal-model :as cm]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.attribution :as attribution]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]))

(def dag {:directed [[:z :x] [:z :y] [:x :y]]})

(defn data [n seed]
  (random/with-stream* seed ::data
    #(mapv (fn [_]
             (let [z (dist/draw (dist/normal 0 1))
                   x (+ z (dist/draw (dist/normal 0 1)))
                   y (+ 1 (* 2 x) (* 3 z) (dist/draw (dist/normal 0 0.5)))]
               {:z z :x x :y y})) (range n))))

(defn effect [fitted]
  (let [noise (zipmap (:order fitted) (repeat 0.0))]
    (- (:y (cm/replay fitted noise {:x 1.0}))
       (:y (cm/replay fitted noise {:x 0.0})))))

(deftest fit-distinguishes-confounding-from-intervention
  (let [rows (data 600 42) fitted (cm/fit dag rows)
        naive (cm/fit {:directed [[:x :y]]} rows)]
    (is (< (Math/abs (- 2 (effect fitted))) 0.08))
    (is (< (Math/abs (- 3.5 (effect naive))) 0.2))
    (is (< (Math/abs (- 3 (get-in fitted [:mechanisms :y :coefficients :z]))) 0.1))
    (is (< (Math/abs (- 0.5 (get-in fitted [:mechanisms :y :sigma]))) 0.05))
    (is (= (cm/simulate fitted 5 {:seed 9}) (cm/simulate fitted 5 {:seed 9})))
    (is (= [7 7 7] (mapv :x (cm/simulate fitted 3 {:seed 9 :interventions {:x 7}}))))))

(deftest abduction-reconstructs-the-incident-and-couples-its-repair
  (let [fitted (cm/fit dag (data 300 1)) incident {:z 2.0 :x 4.0 :y 17.0}
        a (cm/abduct fitted incident)
        factual (cm/replay fitted (:noise a)) repaired (cm/replay fitted (:noise a) {:x 0.0})]
    (doseq [v (:order fitted)]
      (is (< (Math/abs (- (get incident v) (get factual v))) 1e-10)))
    (is (< (Math/abs (- (- (:y factual) (:y repaired)) (* 4 (effect fitted)))) 1e-10))
    (is (= (:z factual) (:z repaired)))))

(deftest qr-solves-rescaled-multivariate-regression
  ;; Independent factorial columns and residuals give an exact oracle,
  ;; including a large-offset parent and six orders of scale difference.
  (let [rows (vec (for [x [-2 -1 1 2] z [-1 1] u [-1 1]]
                    {:x (+ 1e8 x) :z (* 1e-6 z)
                     :y (+ 3 (* 2 x) (* -4 z) (* 0.5 u))}))
        fitted (cm/fit {:directed [[:x :y] [:z :y]]} rows)]
    (is (< (Math/abs (- 2 (get-in fitted [:mechanisms :y :coefficients :x]))) 1e-8))
    (is (< (Math/abs (+ 4e6 (get-in fitted [:mechanisms :y :coefficients :z]))) 0.01))
    (is (< (Math/abs (- 0.5 (get-in fitted [:mechanisms :y :sigma]))) 1e-8))))

(deftest small-measurement-units-do-not-imply-deterministic-noise
  (let [rows (mapv #(update % :y * 1e-12) (data 100 19))
        fitted (cm/fit dag rows)]
    (is (< (Math/abs (- 2 (/ (effect fitted) 1e-12))) 0.2))))

(deftest invalid-and-underidentified-fits-are-refused
  (doseq [[g rows] [[{:nodes #{:x}} []]
                    [{:nodes #{:x}} [{:x ##NaN}]]
                    [{:directed [[:x :y]]} [{:x 1 :y 1} {:x 1 :y 2} {:x 1 :y 3}]]
                    [{:directed [[:x :y]]} [{:x 1 :y 1} {:x 2 :y 2} {:x 3 :y 3}]]
                    [{:directed [[:x :y] [:z :y]]}
                     (mapv (fn [x] {:x x :z (* 2 x) :y (+ x (mod x 2))}) (range 10))]
                    [{:nodes #{:x :y} :bidirected [[:x :y]]} (data 10 9)]]]
    (is (thrown? #?(:clj Exception :cljs js/Error) (cm/fit g rows))))
  (let [fitted (cm/fit dag (data 50 2))]
    (is (thrown? #?(:clj Exception :cljs js/Error) (cm/abduct fitted {:x 1})))
    (is (thrown? #?(:clj Exception :cljs js/Error) (cm/replay fitted {})))
    (is (thrown? #?(:clj Exception :cljs js/Error) (cm/replay fitted {} {:unknown 1})))
    (is (thrown? #?(:clj Exception :cljs js/Error) (cm/simulate fitted 1 {:interventions {:unknown 1}})))))

(deftest held-out-calibration-and-bootstrap-effect-uncertainty
  (let [rows (data 300 77) fitted (cm/fit dag rows)
        opts {:seed 10 :permutations 39}
        report (cm/evaluate fitted (data 180 78) opts)
        boot (cm/bootstrap fitted rows effect {:seed 11 :replicates 60})
        [lo hi] (:interval boot)]
    (is (= report (cm/evaluate fitted (data 180 78) opts)))
    (is (< 0.82 (get-in report [:mechanisms :y :coverage-90]) 0.98))
    (is (< 0.4 (get-in report [:mechanisms :y :rmse]) 0.65))
    (is (= 6 (count (:checks report))))
    (is (every? #(<= (:p-value %) (:adjusted-p-value %) 1) (:checks report)))
    (is (< lo 2 hi))
    (is (= boot (cm/bootstrap fitted rows effect {:seed 11 :replicates 60})))
    (is (= (effect fitted) (:estimate boot)))
    (let [projected (assoc-in fitted [:mechanisms :y :noise-score] #(+ 100 %))
          projected-report (cm/evaluate projected (data 180 78) opts)]
      (is (= (get-in report [:mechanisms :y :coverage-90])
             (get-in projected-report [:mechanisms :y :coverage-90]))))))

(defn nonlinear-data [seed]
  (random/with-stream* seed ::nonlinear
    #(mapv (fn [_]
             (let [x (dist/draw (dist/normal 0 1))]
               {:x x :y (+ (* 3 x x) (dist/draw (dist/normal 0 0.2)))})) (range 160))))

(deftest held-out-quadratic-misspecification-is-detected
  (let [fitted (cm/fit {:directed [[:x :y]]} (nonlinear-data 1))
        r (cm/evaluate fitted (nonlinear-data 2) {:seed 3 :permutations 199})]
    (is (some #(and (= :noise-parent (:kind %)) (= [:y :x] (:nodes %))) (:rejected-checks r)))))

(deftest custom-fitters-retain-structural-coupling
  (let [seen (atom nil)
        fitted (cm/fit {:directed [[:x :y]]} (data 40 7)
                       {:fitters {:y (fn [_ parents _]
                                       (reset! seen parents)
                                       {:family :known-laplace
                                        :distribution (fn [row]
                                                        (is (= #{:x} (set (keys row))))
                                                        (dist/laplace (* 2 (:x row)) 0.5))})}})
        row {:x 1.0 :y 3.0} a (cm/abduct fitted row)]
    (is (= [:x] @seen))
    (is (< (Math/abs (- 5 (:y (cm/replay fitted (:noise a) {:x 2.0})))) 1e-9))))

(deftest custom-mechanisms-refuse-impossible-observations
  (let [fitted (cm/fit {:nodes #{:x}} [{:x 0.2} {:x 0.8}]
                       {:fitters {:x (fn [_ _ _] {:distribution (fn [_] (dist/uniform 0 1))})}})]
    (is (thrown? #?(:clj Exception :cljs js/Error) (cm/abduct fitted {:x 2.0})))
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (cm/evaluate fitted (repeat 4 {:x 2.0}) {:seed 1 :permutations 3})))))

(deftest fitted-model-integrates-with-worlds-attribution-and-repairs
  (let [fitted (cm/fit dag (data 150 4)) incident {:z 0.0 :x 3.0 :y 10.0}
        a (cm/abduct fitted incident) context (ctx/create-execution-context)
        check (fn [result]
                (let [[worlds report repairs] result
                      predicted (cm/replay fitted (:noise a) {:x 0.0})]
                  (is (< (Math/abs (- (:y predicted) (get-in worlds [:worlds :repaired :y]))) 1e-9))
                  (is (< (Math/abs (- (:total report) (:baseline report)
                                      (reduce + (vals (:contributions report))))) 1e-9))
                  (is (= :set-x-zero (:candidate (first repairs))))
                  (is (= 1.0 (:success-probability (first repairs))))))
        task (fn [resolve reject]
               (binding [ec/*execution-context* context]
                 (let [m (cm/model fitted)]
                   ((cf/worlds m a [[:repaired {:x {:do 0.0}}]])
                    (fn [worlds]
                      ((attribution/attribute-noise m a [(zipmap (:order fitted) (repeat 0.0))]
                                                    {:nodes (:order fitted) :score :y})
                       (fn [report]
                         ((attribution/repairs m [a] {:none {} :set-x-zero {:x {:do 0.0}}} #(< (:y %) 6))
                          #(resolve [worlds report %]) reject)) reject)) reject))))]
    #?(:clj (let [p (promise)]
              (task #(deliver p %) #(deliver p %))
              (check (deref p 30000 ::timeout)))
       :cljs (async done (task #(do (check %) (done)) #(do (is false (str %)) (done)))))))
