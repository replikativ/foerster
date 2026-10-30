(ns org.replikativ.foerster.amortized-test
  (:refer-clojure :exclude [await])
  (:require [clojure.set :as set]
            #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is async]])
            [org.replikativ.foerster.amortized :as a]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe factor]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]
            [org.replikativ.spindel.spin.cps :refer [spin]])
  #?(:cljs (:require-macros [org.replikativ.spindel.spin.cps :refer [spin]])))

(defn simulator []
  (spin (let [z (sample (dist/normal 0 1) :id :z)
              x (sample (dist/normal z 1) :id :x)] {:z z :x x})))

(def options {:id :test/v1 :schema [{:name :z :role :target :type :real}
                                    {:name :x :role :observation :type :real}]
              :conditionings [#{:x} #{}]})
(def basic (a/task simulator options))

(def backend
  {:id :test/conditional
   :train (fn [_ rows _] (spin {:examples (count rows)}))
   :sample (fn [_ request {:keys [draws]}]
             (spin (vec (repeat draws [(if (second (:observed request)) (/ (second (:values request)) 2) 0.0)
                                       (second (:values request))]))))
   :freeze identity :thaw (fn [_ data] data)})

(defn run-case [factory check done]
  (binding [engine/*execution-context* (context/create-execution-context)]
    ((factory) (fn [value] (check value) (done))
               (fn [error] (is false (str error)) (done)))))

#?(:clj (defn sync-case [factory check]
          (let [p (promise)]
            (run-case factory #(deliver p %) (fn []))
            (check (deref p 30000 ::timeout)))))

(deftest numeric-and-finite-schema-round-trips
  (doseq [[descriptor value] [[{:type :real :shape [2 2]} [[1.0 2.0] [3.0 4.0]]]
                              [{:type :positive :shape [2]} [0.5 2.0]]
                              [{:type :interval :bounds [-2 3]} 0.25]
                              [{:type :simplex :shape [3]} [0.2 0.3 0.5]]
                              [{:type :finite :values [nil false :other]} false]]]
    (let [task (a/task simulator {:id :types :schema [(merge {:name :z :role :target} descriptor)]})
          v (get-in task [:by-name :z]) result (a/decode v (a/encode v value))]
      (if (= :finite (:type v)) (is (= value result))
          (letfn [(flat [x] (if (sequential? x) (mapcat flat x) [x]))]
            (is (every? #(< (Math/abs %) 1e-10) (map - (flat value) (flat result)))))))))

(deftest support-and-shape-errors-are-explicit
  (doseq [[descriptor value] [[{:type :positive} 0]
                              [{:type :interval :bounds [0 1]} 1]
                              [{:type :simplex :shape [2]} [0.3 0.3]]
                              [{:type :finite :values [false true]} :missing]
                              [{:type :real :shape [2]} [1]]
                              [{:type :real} ##NaN]]]
    (let [task (a/task simulator {:id :invalid :schema [(merge {:name :z :role :target} descriptor)]})]
      (is (thrown? #?(:clj Exception :cljs js/Error) (a/encode-row task {:z value}))))))

(deftest observation-encoders-are-versioned-features
  (let [schema [{:name :z :role :target :type :real}
                {:name :text :role :observation :type :embedding :shape [2]
                 :encoder {:id :test/embed :version "a" :encode (fn [s] [(double (count s)) 1.0])}}]
        task (a/task simulator {:id :text :schema schema})]
    (is (= [0.0 3.0 1.0] (a/encode-row task {:z 0 :text "abc"})))
    (is (nil? (get-in (a/description task) [:schema 1 :encoder :encode])))
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (a/task simulator {:id :bad :schema (assoc-in schema [1 :role] :target)})))
    (let [artifact (a/artifact task backend {}) checkpoint (a/freeze artifact)
          changed (a/task simulator {:id :text :schema (assoc-in schema [1 :encoder :version] "b")})]
      (is (thrown? #?(:clj Exception :cljs js/Error) (a/thaw changed backend checkpoint))))))

(deftest embedding-encoders-can-return-native-numeric-arrays
  (let [task (a/task simulator
                     {:id :native-array :schema [{:name :z :role :target :type :real}
                                                 {:name :text :role :observation :type :embedding :shape [2]
                                                  :encoder {:id :encoder :version "v1"
                                                            :encode (fn [_] #?(:clj (float-array [3 1])
                                                                               :cljs (js/Float32Array. #js [3 1])))}}]})]
    (is (= [0.0 3.0 1.0] (a/encode-row task {:z 0 :text "abc"})))))

(deftest checkpoints-do-not-contain-runtime-closures
  (let [artifact (a/artifact basic backend {:hidden-function identity})]
    (is (thrown? #?(:clj Exception :cljs js/Error) (a/freeze artifact)))))

(deftest masks-and-projected-marginals-do-not-delete-variables
  (let [task (a/task simulator (assoc options :schema (conj (:schema options) {:name :input :role :input :type :real})))]
    (is (= [false true true] (:observed (a/condition task {:x 2 :input 1}))))
    (is (= [false false true] (:observed (a/condition task {:input 1}))))
    (is (thrown? #?(:clj Exception :cljs js/Error) (a/condition task {:x 2})))
    (is (thrown? #?(:clj Exception :cljs js/Error) (a/condition task {:z 1 :input 1})))
    (is (= {:z 3.0} (a/decode-targets task [3.0 2.0 1.0] [:z])))))

(defn check-inference [value]
  (let [[rows posterior restored-posterior] value]
    (is (= 5 (count rows)))
    (is (= :amortized (m/measure-type posterior)))
    (is (nil? (m/log-marginal posterior)))
    (is (= :equal-weight-only (get-in posterior [:diagnostics :ess-meaning])))
    (is (= 2.0 (:mean (m/measure-stats posterior #(get (m/get-value %) :z)))))
    (is (= (mapv (comp m/get-value first) (m/get-particles posterior))
           (mapv (comp m/get-value first) (m/get-particles restored-posterior))))))

(deftest simulator-training-and-checkpoint-compose-as-spins
  (let [factory #(spin
                  (let [rows (await (a/examples basic 5 {:seed 22}))
                        fitted (await (a/train basic backend {:examples rows}))
                        posterior (await (a/infer fitted {:x 4.0} {:draws 4}))
                        restored (a/thaw basic backend (a/freeze fitted))
                        again (await (a/infer restored {:x 4.0} {:draws 4}))]
                    [rows posterior again]))]
    #?(:clj (sync-case factory check-inference)
       :cljs (async done (run-case factory check-inference done)))))

(deftest training-does-not-silently-use-fixed-observations-or-factors
  (let [observed (fn [] (spin (let [z (sample (dist/normal 0 1) :id :z)]
                                {:z z :x (observe (dist/normal z 1) 100.0 :id :x)})))
        scored (fn [] (spin (factor 0.0) {:z 0.0 :x 0.0}))
        factory #(spin
                  (let [rejected-observation? (try (await (a/examples (a/task observed options) 1)) false
                                                   (catch #?(:clj Throwable :cljs :default) _ true))
                        rejected-factor? (try (await (a/examples (a/task scored options) 1)) false
                                              (catch #?(:clj Throwable :cljs :default) _ true))
                        generated (await (a/examples (a/task observed (assoc options :simulate-observed? true)) 2))]
                    [rejected-observation? rejected-factor? generated]))
        check (fn [[observations? factors? rows]]
                (is observations?) (is factors?)
                (is (= 2 (count rows)))
                (is (every? #(not= 100.0 (:x %)) rows)))]
    #?(:clj (sync-case factory check)
       :cljs (async done (run-case factory check done)))))

(deftest causal-reconstruction-is-explicit-and-composable
  (let [task (a/task simulator
                     (assoc options :model simulator
                            :abduct (fn [{:keys [z]} {:keys [x]}]
                                      {:factual {:z z :x x} :noise {:z z :x (- x z)} :unsupported []})))
        factory #(spin (let [artifact (a/artifact task backend {})
                             posterior (await (a/infer artifact {:x 4.0} {:draws 3}))]
                         (await (a/worlds artifact posterior {:x 4.0} [[:changed {:z {:do 0.0}}]]))))
        check (fn [worlds]
                (is (= 3 (count worlds)))
                (is (every? #(= :complete (:status %)) worlds))
                (is (every? #(= {:z 0.0 :x 2.0} (get-in % [:worlds :changed])) worlds)))]
    #?(:clj (sync-case factory check)
       :cljs (async done (run-case factory check done)))))

(deftest simulator-data-needs-neither-densities-nor-inference-guides
  (let [density-calls (atom 0)
        implicit (reify dist/Distribution
                   (-draw [_] 2.0)
                   (-logpdf [_ _] (swap! density-calls inc) (throw (ex-info "No density" {}))))
        model (fn [] (spin {:z (sample implicit :id :z)
                            :x (sample (dist/normal 0 1) :id :x :proposal (dist/normal 1000 0.01))}))
        task (a/task model options)
        factory #(a/examples task 8 {:seed 32})
        check (fn [rows]
                (is (zero? @density-calls))
                (is (every? #(= 2.0 (:z %)) rows))
                (is (every? #(< (Math/abs (:x %)) 10) rows)))]
    #?(:clj (sync-case factory check)
       :cljs (async done (run-case factory check done)))))

(deftest learned-state-can-drive-explicit-predictive-rollouts
  (let [factory #(spin
                  (let [artifact (a/artifact basic backend {})
                        posterior (await (a/infer artifact {:x 4.0} {:draws 2}))]
                    (await (a/predictive posterior
                                         (fn [{:keys [z]}] (spin (sample (dist/normal z 0.00001) :id :future)))
                                         8 {:seed 45}))))
        check (fn [rollouts]
                (is (= 8 (count rollouts)))
                (is (every? #(< (Math/abs (- (:value %) 2.0)) 0.001) rollouts)))]
    #?(:clj (sync-case factory check)
       :cljs (async done (run-case factory check done)))))

(deftest a-target-block-proposal-is-not-treated-as-a-joint-simulator
  (let [b (block/block {:block/id :target :block/latents [{:name :z :shape [1] :support :real}]
                        :block/target :complete-conditional}
                       {:log-density (fn [_ _] 0.0) :value+grad (fn [_ _] [0.0 [0.0]])
                        :sample (fn [_] [0.0])})
        model (fn [] (spin {:z (sample (block/block-dist b {}) :id :z) :x 0.0}))
        task (a/task model (assoc-in options [:schema 0 :shape] [1]))
        factory #(spin (try (await (a/examples task 1)) false
                            (catch #?(:clj Throwable :cljs :default) _ true)))
        check (fn [rejected?] (is rejected?))]
    #?(:clj (sync-case factory check)
       :cljs (async done (run-case factory check done)))))

(deftest held-out-rows-and-explicit-posterior-assessment
  (let [captured (atom nil)
        recording (assoc backend :train (fn [_ rows opts]
                                          (reset! captured {:rows rows :validation (:validation-rows opts)})
                                          (spin {:examples (count rows)})))
        factory (fn [] (spin
                        (let [fitted (await (a/train basic recording {:simulations 4 :validation-simulations 3 :seed 89}))
                              first-record @captured
                              expected (await (a/examples basic 3 {:seed 93}))
                              passed (await (a/validate fitted (fn [_] (spin {:scope :posterior :passed? true :checks :declared-test}))))
                              failed (await (a/validate fitted (fn [_] (spin {:scope :posterior :passed? false :checks :declared-test}))))
                              resumed (await (a/resume passed {:examples expected :validation-simulations 0}))]
                          {:record @captured :first-record first-record :expected (mapv #(a/encode-row basic %) expected)
                           :initial fitted :passed passed :failed failed :resumed resumed})))
        check (fn [{:keys [record first-record expected initial passed failed resumed]}]
                (is (= 4 (count (:rows first-record))))
                (is (= expected (:validation first-record)))
                (is (empty? (set/intersection (set (:rows first-record)) (set expected))))
                (is (= expected (:rows record)))
                (is (= :unvalidated (:deployment-status initial)))
                (is (= :validation-passed (:deployment-status passed)))
                (is (= :validation-failed (:deployment-status failed)))
                (is (= :unvalidated (:deployment-status resumed))))]
    #?(:clj (sync-case factory check)
       :cljs (async done (run-case factory check done)))))

(deftest fully-observed-training-masks-cannot-create-a-zero-loss-fit
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (a/task simulator {:id :invalid-zero-loss
                                  :schema [{:name :z :role :variable :type :real}]
                                  :conditionings [#{:z}]}))))
