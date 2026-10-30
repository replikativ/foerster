(ns org.replikativ.foerster.nested-filter-contract-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.smc2 :as smc2]
            [org.replikativ.foerster.scope :as scope]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- run [operation]
  (let [result (promise)]
    (operation #(deliver result [:ok %]) #(deliver result [:error %]))
    (let [[status value] (deref result 10000 [::timeout nil])]
      (case status
        :ok value
        :error (throw value)
        (throw (ex-info "Filter operation timed out" {}))))))

(defn- with-root [f]
  (let [root (context/create-execution-context)]
    (try
      (binding [engine/*execution-context* root] (f))
      (finally (context/stop-context! root)))))

(deftest smc2-rejects-a-parameter-prior-after-an-inner-latent
  (with-root
    (fn []
      (let [model (spin (let [z (sample (d/flip 0.5) :id :z)
                              theta (sample (d/flip (if z 0.9 0.1)) :id :theta)]
                          (sample (d/flip (if theta 0.8 0.2)) :id :y :stream true)
                          theta))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"parameters must precede"
                              (run (smc2/smc2 model {:params #{:theta} :n-theta 2 :n-x 2 :seed 17}))))))))

(deftest unsupported-inner-filter-options-fail-before-simulation
  (with-root
    (fn []
      (let [executed (atom 0)
            model (spin (swap! executed inc)
                        (sample (d/flip 0.5) :id :theta)
                        (sample (d/flip 0.5) :id :y :stream true))]
        (doseq [[option value type] [[:world-policy :fork ::smc2/unsupported-world-policy]
                                     [:world-opts {:systems :all} ::smc2/unsupported-world-policy]
                                     [:anchors :all ::smc2/nonforkable-inner-filter]
                                     [:retained {:theta true} ::smc2/nonforkable-inner-filter]
                                     [:copy? true ::smc2/nonforkable-inner-filter]]]
          (testing (str option)
            (let [error (try
                          (run (smc2/smc2 model (assoc {:params #{:theta} :n-theta 2 :n-x 2} option value)))
                          nil
                          (catch clojure.lang.ExceptionInfo e e))]
              (is (= type (:type (ex-data error))))
              (is (zero? @executed) "unsupported options are rejected before executing application code"))))))))

(deftest dependent-parameters-are-allowed-within-the-prior-prefix
  (with-root
    (fn []
      (let [model (spin (let [a (sample (d/flip 0.5) :id :a)
                              b (sample (d/flip (if a 0.9 0.1)) :id :b)]
                          (sample (d/flip (if b 0.8 0.2)) :id :y :stream true)
                          {:a a :b b}))
            step (run (smc2/smc2 model {:params #{:a :b} :n-theta 3 :n-x 2 :seed 19}))]
        (try
          (is (false? (:done? step)))
          (is (= 3 (count (m/get-particles (:measure step)))))
          (let [final (run ((:push step) true))]
            (is (:done? final))
            (is (every? #(= #{:a :b} (set (keys (m/get-value (first %)))))
                        (m/get-particles (:measure final)))))
          (finally (run (:close-operation step))))))))

(deftest a-stream-step-cannot-consume-two-observations
  (with-root
    (fn []
      (let [model (spin (let [a (sample (d/normal 0 1) :id :y0 :stream true)
                              b (sample (d/normal 0 1) :id :y1 :stream true)]
                          [a b]))
            start (run (smc/stream model 2 {:seed 23}))]
        (try
          (let [next-step (run ((:push start) 1.0))]
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stale, closed or already in flight"
                                  (run ((:push start) 99.0))))
            (let [final (run ((:push next-step) 2.0))]
              (is (:done? final))
              (is (= [[1.0 2.0] [1.0 2.0]]
                     (mapv (comp m/get-value first) (m/get-particles (:measure final)))))))
          (finally (run (:close-operation start))))))))

(deftest proposal-control-flow-cannot-break-the-parameter-prefix
  (with-root
    (fn []
      (let [model (spin (let [a (sample (d/uniform 0 1) :id :a)
                              _ (when (> a 0.5) (sample (d/flip 0.5) :id :z))
                              b (sample (d/uniform 0 1) :id :b)]
                          (sample (d/flip 0.5) :id :y :stream true)
                          [a b]))]
        (with-redefs-fn {#'smc2/prior-draw (fn [& _]
                                             (fn [resolve _] (resolve {:a 0.25 :b 0.25})))
                         #'smc2/random-walk (fn [_]
                                              (fn [_] {:theta {:a 0.75 :b 0.25} :log-ratio 0.0}))}
          #(let [step (run (smc2/smc2 model {:params #{:a :b} :n-theta 2 :n-x 2
                                             :ess-target 2.0 :moves 1 :seed 34}))]
             (try
               (is (thrown-with-msg? clojure.lang.ExceptionInfo #"parameters must precede"
                                     (run ((:push step) true))))
               (finally (run (:close-operation step))))))))))

(deftest a-stream-barrier-cannot-hide-unreached-parameters
  (with-root
    (fn []
      (let [model (spin (sample (d/flip 0.5) :id :a)
                        (sample (d/flip 0.5) :id :y :stream true)
                        (sample (d/flip 0.5) :id :b))]
        (with-redefs-fn {#'smc2/prior-draw (fn [& _]
                                             (fn [resolve _] (resolve {:a true :b true})))}
          #(is (thrown-with-msg? clojure.lang.ExceptionInfo #"first inner stream barrier"
                                 (run (smc2/smc2 model {:params #{:a :b} :n-theta 1 :n-x 1 :seed 35})))))))))

(deftest failed-initialization-and-push-await-owned-filter-cleanup
  (doseq [failure-stage [:initial :push]]
    (testing (name failure-stage)
      (let [closing (promise) finish-close (atom nil) outcome (promise)
            failure (ex-info "owned inner failure" {:type ::inner-failure})
            measure (assoc (m/empirical [[(m/sample-particle :value {:theta {:value 0.25 :log-prob 0.0}}) 0.0]])
                           :log-normalizer 0.0)]
        (with-redefs-fn {#'smc2/prior-draw (fn [& _]
                                             (fn [resolve _] (resolve {:theta 0.25})))
                         #'smc/stream (fn [_ _ opts]
                                        (scope/acquire! (:inference-scope opts) (constantly :filter)
                                                        (fn [_]
                                                          (fn [resolve _]
                                                            (reset! finish-close resolve)
                                                            (deliver closing true))))
                                        (fn [resolve reject]
                                          (if (= :initial failure-stage)
                                            (reject failure)
                                            (resolve {:measure measure :done? false :close (fn [])
                                                      :push (fn [_] (fn [_ reject] (reject failure)))}))))}
          #(let [start (smc2/smc2 nil {:params #{:theta} :n-theta 1 :n-x 1 :seed 36})
                 op (if (= :initial failure-stage) start ((:push (run start)) :observation))]
             (op (fn [v] (deliver outcome [:ok v])) (fn [e] (deliver outcome [:error e])))
             (is (= true (deref closing 10000 ::timeout)))
             (is (not (realized? outcome)) "failure is withheld until owned close finishes")
             (@finish-close nil)
             (let [[status error] (deref outcome 10000 [::timeout nil])]
               (is (= :error status))
               (is (identical? failure error)))))))))
