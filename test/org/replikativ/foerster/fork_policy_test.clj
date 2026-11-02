(ns org.replikativ.foerster.fork-policy-test
  "`:world-policy :fork` beyond savepoint SMC: one frozen fork of the
  caller's world per inference, a world and scope of its own for every run
  (sweep, chain, particle, iteration), no incremental reuse under :fork, and
  refusals where a method has no worlds."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.executor :as executor]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.world.scope :as world-scope]))

(defmacro ^:private capturing-scopes
  "Run `body` with every world scope created meanwhile collected in `scopes`."
  [[scopes] & body]
  `(let [~scopes (atom [])
         create# world-scope/create]
     (with-redefs [world-scope/create (fn [opts#]
                                        (let [s# (create# opts#)]
                                          (swap! ~scopes conj s#)
                                          s#))]
       ~@body)))

(defn- settled? [scopes]
  (every? #(and (= :discarded (:status @%)) (empty? (:handles @%))) scopes))

(defmacro ^:private in-root
  "Run `body` with a fresh caller world bound, stopped afterwards."
  [& body]
  `(let [root# (context/create-execution-context)]
     (try
       (binding [ec/*execution-context* root#] ~@body)
       (finally (context/stop-context! root#)))))

(defn- failure [run]
  (deref (spin (try (await (run)) :unexpected-success
                    (catch Throwable error error)))
         60000 ::timed-out))

;; -----------------------------------------------------------------------------
;; Compiled paths have no worlds
;; -----------------------------------------------------------------------------

(def ^:private normal-mean
  (a/model [ys]
           (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
             (loop [i 0]
               (when (< i (count ys))
                 (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
                 (recur (inc i))))
             mu)))

(deftest compiled-rmh-refuses-world-options
  (in-root
   (doseq [opts [{:world-policy :fork} {:world-opts {}} {:authority :ledger :grant {:tokens 1}}]]
     (let [error (try (inference/infer (normal-mean [0.5 1.0])
                                       (merge {:method :rmh :compiled? true :iterations 10 :chains 1} opts))
                      nil
                      (catch clojure.lang.ExceptionInfo e e))]
       (is (= ::inference/invalid-world-policy (:type (ex-data error))) (pr-str opts))))
   (testing ":world-policy :fresh is what compiled code does anyway"
     (is (some? (inference/infer (normal-mean [0.5 1.0])
                                 {:method :rmh :compiled? true :iterations 10 :chains 1
                                  :world-policy :fresh}))))))

;; -----------------------------------------------------------------------------
;; Batched SMC in canonical worlds
;; -----------------------------------------------------------------------------

(deftest batched-smc-runs-in-canonical-worlds
  (let [exec (executor/thread-pool-executor {:threads 4})]
    (try
      (in-root
       (let [log (sa/atom [:root])
             model (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                           (observe (dist/normal x 1.0) 0.5 :id :y)
                           (observe (dist/normal x 1.0) 0.7 :id :z)
                           (swap! log conj :mine)
                           [x @log]))]
         (capturing-scopes [scopes]
                           (let [posterior @(spin (await (inference/smc-infer
                                                          model 400 {:world-policy :fork :batch 16 :executor exec
                                                                     :resample-threshold 1.0 :seed 3})))
                                 values (mapv measure/get-value (measure/get-contexts posterior))
                                 ws (mapv #(Math/exp (second %)) (measure/get-particles posterior))
                                 mean (/ (reduce + (map * (map first values) ws)) (reduce + ws))]
                             (is (= 400 (count values)))
                             (is (every? #{[:root :mine]} (map second values))
                                 "every particle sees the caller's state and its own write only")
                             (is (= [:root] @log) "nothing reaches the caller")
                             (is (< (Math/abs (- mean 0.4)) 0.15) (str mean " vs 0.4"))
                             (is (some #(<= (:size %) 16) (:history posterior)) "resampled in batches")
                             (is (every? #(= :discarded (:fork/status %)) (measure/world-descriptors posterior)))
                             (is (settled? @scopes))))))
      (finally (.close ^java.lang.AutoCloseable exec)))))
