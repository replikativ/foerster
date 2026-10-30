(ns org.replikativ.foerster.amortized-compiler-test
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.amortized.compiler :as c]
            [org.replikativ.foerster.amortized :as a]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn ordinary-simulator []
  (spin (let [z (sample (d/normal 0 1)) x (sample (d/normal z 1))]
          {:latent z :reading x})))

(defn- run [factory]
  (binding [engine/*execution-context* (context/create-execution-context)] @(factory)))

(deftest existing-defn-needs-no-manual-schema-or-mask
  (let [task (c/compile #'ordinary-simulator)
        rows (run #(a/examples task 3))]
    (is (= #{:latent :reading} (set (map :name (:schema task)))))
    (is (= [[:latent :reading]] (:attention-edges task)))
    (is (= :any (:conditionings task)))
    (is (= 3 (count rows)))
    (is (every? #(= #{:latent :reading} (set (keys %))) rows))
    (is (= [true false] (:observed (a/condition task {:latent 1.0}))))))

(deftest co-parents-are-connected-for-posterior-inference
  (let [model (c/model [] (let [a (sample (d/normal 0 1) :id :a)
                                b (sample (d/normal 0 1) :id :b)
                                y (sample (d/normal (+ a b) 1) :id :y)] [a b y]))
        task (c/compile model)]
    (is (= #{#{:a :b} #{:a :y} #{:b :y}} (set (map set (:attention-edges task)))))
    (is (= :factor-scope-cliques (get-in task [:semantics :attention])))))

(deftest public-core-effects-are-recognized
  (let [model (c/model [] (let [x (inference/sample (d/normal 0 1) :id :x)
                                y (inference/sample (d/normal x 1) :id :y)] {:x x :y y}))
        task (c/compile model)]
    (is (= [[:x :y]] (:attention-edges task)))
    (is (= 2 (count (run #(a/examples task 2)))))))

(deftest plate-instances-infer-hierarchical-identities
  (let [model (c/model [n] (loop [i 0 xs []]
                             (if (= i n) xs
                                 (recur (inc i) (conj xs (sample (d/normal 0 1) :id [:x i]))))))
        task (c/compile model {:args [3]})]
    (is (= [[:x 0] [:x 1] [:x 2]] (mapv :name (:schema task))))
    (is (= #{[:x 0] [:x 1] [:x 2]} (set (keys (first (run #(a/examples task 1)))))))))

(deftest types-are-derived-from-declared-distributions
  (let [model (c/model [] (let [i (sample (d/flip 0.3) :id :i)
                                x (sample (d/exponential 2.0) :id :x)
                                p (sample (d/beta 2.0 3.0) :id :p)] [i x p]))
        task (c/compile model)]
    (is (= [:finite :positive :interval] (mapv :type (:schema task))))
    (is (= [false true] (get-in task [:schema 0 :values])))))

(deftest variable-trace-structure-is-not-guessed-from-one-run
  (let [model (c/model [] (if (sample (d/flip 0.5) :id :branch)
                            (sample (d/normal 0 1) :id :x) 0.0))]
    (is (thrown? Exception (c/compile model)))))

(deftest opaque-effects-get-conservative-connectivity
  (let [opaque (fn [x] (* x x))
        model (c/model [] (let [x (sample (d/normal 0 1) :id :x)
                                y (sample (d/normal 0 1) :id :y)
                                z (sample (d/normal (opaque x) 1) :id :z)] [x y z]))
        task (c/compile model)]
    (is (= 3 (count (:attention-edges task))))
    (is (= :dense-conservative (get-in task [:semantics :attention])))))

(deftest causal-noise-reconstruction-is-derived-after-an-explicit-declaration
  (let [model (c/model [] (let [z (sample (d/normal 0 1) :id :z)
                                x (observe (d/normal z 1) 100.0 :id :x)] {:z z :x x}))
        task (c/compile model {:causal? true})
        backend {:id :oracle :sample (fn [_ request {:keys [draws]}]
                                       (spin (vec (repeat draws [2.0 (second (:values request))]))))}
        worlds (run #(spin (let [artifact (a/artifact task backend {})
                                 posterior (await (a/infer artifact {:x 4.0} {:draws 3 :targets [:z]}))]
                             (await (a/worlds artifact posterior {:x 4.0} [[:changed {:z {:do 0.0}}]])))))]
    (is (= 3 (count worlds)))
    (is (every? #(= {:z 0.0 :x 2.0} (get-in % [:worlds :changed])) worlds))))

(deftest marginal-query-retains-other-latents-for-causal-reconstruction
  (let [model (c/model [] (let [z (sample (d/normal 0 1) :id :z)
                                x (sample (d/normal z 1) :id :x)
                                y (sample (d/normal (+ (* 2 x) z) 1) :id :y)] {:z z :x x :y y}))
        task (c/compile model {:causal? true})
        backend {:id :oracle :sample (fn [_ _ {:keys [draws]}] (spin (vec (repeat draws [2.0 4.0 7.0]))))}
        [posterior worlds] (run #(spin (let [artifact (a/artifact task backend {})
                                             posterior (await (a/infer artifact {:y 7.0} {:draws 2 :targets [:z]}))
                                             worlds (await (a/worlds artifact posterior {:y 7.0} [[:changed {:z {:do 0.0}}]]))]
                                         [posterior worlds])))]
    (is (= {:z 2.0} (m/get-value (ffirst (m/get-particles posterior)))))
    (is (every? #(= {:z 0.0 :x 2.0 :y 1.0} (get-in % [:worlds :changed])) worlds))))

(deftest trace-less-predictive-replay-does-not-silently-draw-the-prior
  (let [posterior (m/empirical [[(m/sample-particle {:z 100.0} nil) 0.0]])]
    (is (thrown? Exception
                 (run #(inference/predictive (ordinary-simulator) posterior 1))))))
