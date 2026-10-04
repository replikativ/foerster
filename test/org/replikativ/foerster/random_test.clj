(ns org.replikativ.foerster.random-test
  "A seeded run draws the same numbers however its particles and chains are
  scheduled: every draw made in a world reads a stream keyed by the world's
  seed and what is drawn."
  (:require [clojure.walk :as walk]
            [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :refer [run-infer weighted-values]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.involutive :as inv]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.spindel.engine.executor :as executor]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.foerster.dist :as dist]))

(defn- model []
  (spin
   (let [x (sample (dist/normal 0 1))]
     (observe (dist/normal x 1) 2.0)
     (let [z (sample (dist/normal 0 1))]
       (observe (dist/normal z 1) -2.0)
       [x z]))))

(defn- runs
  "`n` runs of `make` under `seed` on a 4-thread executor."
  [n seed make]
  (let [exec (executor/thread-pool-executor {:threads 4})]
    (try
      (vec (repeatedly n #(weighted-values (run-infer seed (fn [] (make exec))))))
      (finally (.close ^java.lang.AutoCloseable exec)))))

(deftest parallel-chains-are-reproducible
  (let [make (fn [exec] (infer/kernel-infer (model)
                                            (k/random-walk-mh-kernel 300 {:step-size 0.5 :samples :all :burn 50})
                                            4 {:executor exec}))
        [a b c] (runs 3 13 make)]
    (is (= a b c) "the same seed, the same chains")
    (is (not= a (first (runs 1 14 make))) "another seed, other chains")))

(deftest parallel-particles-are-reproducible
  (let [make (fn [exec] (smc/smc (model) 200 {:executor exec}))
        [a b c] (runs 3 21 make)]
    (is (= a b c))
    (is (not= a (first (runs 1 22 make))))))

(deftest parallel-ipmcmc-is-reproducible
  ;; its nodes are SMC runs in parallel: each has its own seed and draws its
  ;; resampling from its own streams
  (let [make (fn [exec] (infer/ipmcmc-infer (model) 20 6 {:num-nodes 4 :executor exec}))
        [a b c] (runs 3 41 make)]
    (is (= a b c))))

(deftest parallel-pimh-is-reproducible
  (let [make (fn [exec] (smc/pimh (model) 20 10 {:executor exec}))
        [a b] (runs 2 31 make)]
    (is (= a b))))

;; Moves draw from their worlds' streams, not from the process generator:
;; other draws interleaved between the moves (another computation's, on
;; another thread) do not change them.

(defn- await-cps [operation]
  (let [result (promise)]
    (operation #(deliver result [:ok %]) #(deliver result [:error %]))
    (let [[tag v] (deref result 20000 [:error (ex-info "timed out" {})])]
      (if (= :ok tag) v (throw v)))))

(defn- named-model []
  (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
          (observe (dist/normal x 1.0) 2.0 :id :y)
          x)))

(defn- moves
  "The values of :x along `n` moves `(step t)` from a simulated trace under
  `seed`; with `interleave?` a process draw precedes every move."
  [seed n step interleave?]
  (random/set-seed! seed)
  (let [root (context/create-execution-context)
        model (binding [ec/*execution-context* root] (named-model))]
    (loop [t (await-cps (gfi/simulate model)) i 0 xs []]
      (if (= i n)
        (do (await-cps (gfi/close! t)) xs)
        (do (when interleave? (random/uniform01))
            (let [{t' :trace} (await-cps (step t))]
              (recur t' (inc i) (conj xs (get-in t' [:trace/entries :x :value])))))))))

(deftest moves-draw-from-world-streams
  (testing "gfi/mh"
    (let [step #(gfi/mh % #{:x})]
      (is (= (moves 5 40 step false) (moves 5 40 step true)))))
  (testing "involutive/step"
    (let [q (dist/normal 0.0 0.8)
          move {:propose (fn [_] (let [u (dist/draw q)] {:aux u :log-q (dist/logpdf q u)}))
                :log-q (fn [_ u] (dist/logpdf q u))
                :involution (fn [{x :x} u] {:choices {:x (+ x u)} :aux (- u) :log-jacobian 0.0})}
          step #(inv/step % move)
          xs (moves 6 40 step false)]
      (is (= xs (moves 6 40 step true)))
      (is (< 5 (count (distinct xs))) "repeated moves from one state propose anew"))))

(deftest abduction-draws-from-the-factual-world
  (let [noise (fn [interleave?]
                (random/set-seed! 9)
                (let [root (context/create-execution-context)
                      model (binding [ec/*execution-context* root]
                              (spin (sample (dist/flip 0.3) :id :c)))
                      t (await-cps (gfi/simulate model))]
                  (when interleave? (random/uniform01))
                  (let [[n] (cf/noise-of t)]
                    (await-cps (gfi/close! t))
                    n)))]
    (is (= (noise false) (noise true)))))

(deftest generator-preserves-uuid-word-layout
  (doseq [seed [0 1 -1 1.5 :seed [42 :x] {:a [1 2.0]} '(1 :x 2.0) #{1 2.0}
                (first {:a 2.0})]]
    (let [canonical (walk/postwalk
                     (fn [v] (if (and (number? v) (== v (Math/floor v)))
                               (str (long v)) v))
                     seed)
          _ (is (= canonical (#'org.replikativ.foerster.random/canonical seed)))
          uuid (org.replikativ.spindel.engine.hash/content-hash
                [:org.replikativ.foerster.random/stream
                 canonical])
          hex (clojure.string/replace (str uuid) "-" "")
          words (mapv #(Long/parseLong (subs hex (* 8 %) (* 8 (inc %))) 16)
                      (range 4))
          expected (if (every? zero? words) [1 0 0 0] words)
          g (random/generator seed)]
      (is (= expected @g))
      (loop [state expected i 0]
        (when (< i 100)
          (let [[out next-state] (#'org.replikativ.foerster.random/step state)]
            (is (= out (random/next-u32! g)))
            (is (= next-state @g))
            (recur next-state (inc i))))))))
