(ns org.replikativ.foerster.amortized-integration-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.amortized :as a]
            [org.replikativ.foerster.amortized.compiler :as c]
            [org.replikativ.foerster-raster.amortized :as gsdm]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]))

(def program
  (c/model [] (let [x (sample (d/normal 0 1) :id :x)
                    y (sample (d/normal x 1) :id :y)
                    z (sample (d/normal y 1) :id :z)] {:x x :y y :z z})))
(def task (c/compile program {:id :integration/v1}))
(def compiled (delay (gsdm/prepare task {:emb-dim 8 :n-layers 1})))

(defn values [posterior] (mapv (comp m/get-value first) (m/get-particles posterior)))

(deftest unsupported-device-selection-does-not-silently-use-cpu
  (is (thrown? Exception (gsdm/prepare task {:target-device :ze:0})))
  (is (thrown? Exception (gsdm/prepare task {:dtype :float}))))

(deftest source-training-sampling-and-data-checkpoints
  (binding [engine/*execution-context* (context/create-execution-context)]
    (let [backend (gsdm/backend {:compiled @compiled :emb-dim 8 :n-layers 1})
          fitted @(a/train task backend {:simulations 32 :updates 20 :seed 23})
          checkpoint (a/freeze fitted)
          ;; Round-trip through EDN, not merely an in-memory identity.
          restored (a/thaw task backend (clojure.edn/read-string (pr-str checkpoint)))
          first @(a/infer fitted {:z 1.0} {:targets [:x :y] :draws 8 :seed 25})
          second @(a/infer restored {:z 1.0} {:targets [:x :y] :draws 8 :seed 25})
          marginal @(a/infer restored {:z 1.0} {:targets [:x] :draws 8 :seed 25})]
      (is (= 8 (count (values first))))
      (is (= (values first) (values second)))
      (is (= (mapv :x (values second)) (mapv :x (values marginal))))
      (is (every? #(= #{:x} (set (keys %))) (values marginal)))
      (is (= :amortized (m/measure-type first)))
      (is (nil? (m/log-marginal first)))
      (is (thrown? Exception (a/thaw task backend (update-in checkpoint [:payload :flat] dissoc :embed-We)))))))

(deftest engine-reuse-is-qualified-by-connectivity-not-only-rv-count
  (let [different (assoc task :attention-edges nil)]
    (is (thrown? Exception
                 (gsdm/prepare different {:compiled @compiled :emb-dim 8 :n-layers 1})))))

(deftest training-outcomes-progress-and-held-out-selection
  (binding [engine/*execution-context* (context/create-execution-context)]
    (let [events (atom [])
          backend (gsdm/backend {:compiled @compiled :emb-dim 8 :n-layers 1})
          fitted @(a/train task backend {:simulations 32 :validation-simulations 8
                                         :updates 4 :validation-every 2 :validation-cases 8
                                         :on-progress #(swap! events conj %)})
          receipt (get-in fitted [:payload :training])
          stopped @(a/train task backend {:simulations 32 :validation-simulations 8
                                          :updates 4 :validation-every 1 :validation-cases 8
                                          :patience 1 :min-delta 1e12})
          timed @(a/train task backend {:simulations 32 :validation-simulations 8
                                        :updates 4 :validation-cases 8 :max-seconds 1e-12})]
      (is (= :budget-exhausted (:status receipt)))
      (is (= :unverified (:fit-quality receipt)))
      (is (= 8 (:validation-examples receipt)))
      (is (= 2 (count (:history receipt))))
      (is (= :budget-exhausted (:status (last @events))))
      (is (<= (:selected-validation-loss receipt) (:initial-validation-loss receipt)))
      (is (= :validation-plateau (get-in stopped [:payload :training :status])))
      (is (= 1 (get-in stopped [:payload :training :updates])))
      (is (= :time-limit (get-in timed [:payload :training :status])))
      (is (= 0 (get-in timed [:payload :training :updates]))))))

(deftest checkpoint-resume-preserves-adam-and-step-local-randomness
  (binding [engine/*execution-context* (context/create-execution-context)]
    (let [backend (gsdm/backend {:compiled @compiled :emb-dim 8 :n-layers 1})
          options {:simulations 32 :validation-simulations 8 :validation-cases 8
                   :restore-best? false :seed 123 :rate 0.0001}
          full @(a/train task backend (assoc options :updates 10))
          part @(a/train task backend (assoc options :updates 4))
          restored (a/thaw task backend (clojure.edn/read-string (pr-str (a/freeze part))))
          resumed @(a/resume restored (assoc options :updates 6))
          full-data (:payload (a/freeze full))
          resumed-data (:payload (a/freeze resumed))]
      (is (= (:flat full-data) (:flat resumed-data)))
      (is (= (:optimizer full-data) (:optimizer resumed-data)))
      (is (= 10 (get-in resumed-data [:optimizer :step])))
      (is (thrown? Exception (a/thaw task backend (assoc-in (a/freeze part) [:payload :optimizer :step] -1))))
      (is (thrown? Exception (a/thaw task backend (assoc-in (a/freeze part) [:payload :optimizer :adam :m :embed-We] [Double/NaN]))))
      (is (thrown? Exception @(a/resume restored (assoc options :updates 1 :seed 124)))))))

(deftest numerical-failure-is-visible-and-refused-by-inference
  (binding [engine/*execution-context* (context/create-execution-context)]
    (let [broken (assoc-in @compiled [:training :train-fn] (fn [& _] Double/NaN))
          backend (gsdm/backend {:compiled broken :emb-dim 8 :n-layers 1})
          fitted @(a/train task backend {:simulations 8 :validation-simulations 0 :updates 2})]
      (is (= :numerical-failure (get-in fitted [:payload :training :status])))
      (is (= :failed (get-in fitted [:payload :training :fit-quality])))
      (is (thrown? Exception (a/infer fitted {:z 1.0}))))))

(deftest state-space-reference-agrees-with-direct-gaussian-conditioning
  (let [reference (requiring-resolve 'org.replikativ.foerster.amortized-scaling/reference)
        one (first (reference 1 {[:y 0] 1.0}))
        prior (reference 2 {})
        ;; Direct 2x2 Gaussian conditioning with covariance P and noise .25 I.
        a 0.36 b 0.288 c 0.5904 det (- (* (+ a 0.25) (+ c 0.25)) (* b b))
        inv00 (/ (+ c 0.25) det) inv01 (/ (- b) det) inv11 (/ (+ a 0.25) det)
        k00 (+ (* a inv00) (* b inv01)) k01 (+ (* a inv01) (* b inv11))
        k10 (+ (* b inv00) (* c inv01)) k11 (+ (* b inv01) (* c inv11))
        result (reference 2 {[:y 0] 1.0 [:y 1] -0.5})]
    (is (< (Math/abs (- (:mean one) (/ 0.36 0.61))) 1e-12))
    (is (< (Math/abs (- (:variance one) (/ (* 0.36 0.25) 0.61))) 1e-12))
    (is (= [0.36 0.5904] (mapv :variance prior)))
    (is (< (Math/abs (- (:mean (result 0)) (+ k00 (* -0.5 k01)))) 1e-12))
    (is (< (Math/abs (- (:mean (result 1)) (+ k10 (* -0.5 k11)))) 1e-12))
    (is (< (Math/abs (- (:variance (result 0)) (- a (+ (* k00 a) (* k01 b))))) 1e-12))
    (is (< (Math/abs (- (:variance (result 1)) (- c (+ (* k10 b) (* k11 c))))) 1e-12))))
